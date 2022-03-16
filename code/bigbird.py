import argparse
import gc
import math
import os
from argparse import Namespace
from datetime import timedelta
from multiprocessing import cpu_count
from typing import List

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import OrdinalEncoder
from torch import nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from transformers import (AdamW, AutoConfig, AutoModel, AutoModelWithLMHead,
                          AutoTokenizer, get_linear_schedule_with_warmup, BertModel, BigBirdTokenizer, BertConfig)
import wandb
# In[2]:

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


seed_everything(314)

# In[3]:
from typing import Any, List, Optional, Tuple, Union, Dict

class MCC(Metric):
 
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.flatten().long())
        self.target.append(target.flatten().long())

    def compute(self) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:

        preds = torch.cat(self.preds, dim=0).cpu().numpy()
        target = torch.cat(self.target, dim=0).cpu().numpy()
        return matthews_corrcoef(preds, target), preds, target


class ModelBigBird(nn.Module):

    def __init__(self, model_name: str, bertconfig: BertConfig, drop_mult: float, use_special_classifier:str):

        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.bert.config = bertconfig
        self.dropout_mult = drop_mult
        self.dropout = nn.Dropout(self.dropout_mult)

        sizes_classifier = [self.bert.config.hidden_size*4, self.bert.config.hidden_size, int(self.bert.config.hidden_size//2), 1]

        if use_special_classifier == 'ln': 
            self.classifier = nn.Sequential(
                self.dropout,
                nn.Linear(sizes_classifier[0], sizes_classifier[1]),
                nn.LayerNorm(sizes_classifier[1]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[1], sizes_classifier[2]),
                nn.LayerNorm(sizes_classifier[2]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[2], sizes_classifier[3]),
                nn.LayerNorm(sizes_classifier[3]),
            )

        elif use_special_classifier == 'bn': 
            self.classifier = nn.Sequential(
                self.dropout,
                nn.Linear(sizes_classifier[0], sizes_classifier[1]),
                nn.BatchNorm1d(sizes_classifier[1]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[1], sizes_classifier[2]),
                nn.BatchNorm1d(sizes_classifier[2]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[2], sizes_classifier[3]),
                nn.BatchNorm1d(sizes_classifier[3]),
            )

        elif use_special_classifier == 'none': 
            self.classifier = nn.Sequential(
                self.dropout,
                nn.Linear(sizes_classifier[0], sizes_classifier[1]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[1], sizes_classifier[2]),
                Mish(),
                self.dropout,
                nn.Linear(sizes_classifier[2], sizes_classifier[3])
            )

    #input_ids, token_type_ids, attention_masks
    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor, device='cuda'):
        #out_bert = self.bert(input_ids, attention_masks)[0][:,0,:]
        out_bert = self.bert(input_ids, attention_masks, output_hidden_states=True)
        hidden_states = out_bert[1]
        h12 = hidden_states[-1][:,0]
        h11 = hidden_states[-2][:,0]
        h10 = hidden_states[-3][:,0]
        h09 = hidden_states[-4][:,0]
        concat_hidden = torch.cat([h09, h10, h11, h12], axis=-1)
    
        out = self.classifier(concat_hidden)
        return out

    def freeze_bert_encoder(self):
        print('Freezing all bert encoder')
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        print('Unreezing all bert encoder')
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        print('Unfreezing bert encoder last layers')
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        print('Unfreezing bert encoder last pooler layer')
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                print(name)
                param.requires_grad = True



class EncodeCollateFn:

    def slice_text(self, text):
        split = text.split()
        size = len(split)
        if size > self.max_tokens:
            new_text = split[:self.max_tokens//2] + split[-self.max_tokens//2:]
            text = ' '.join(new_text)
        return text

    def __init__(self, tokenizer: AutoTokenizer, max_input_length=7680):

        self.tokenizer = tokenizer
        self.max_tokens = max_input_length

    def __call__(self, batch):

        documents = [self.slice_text(x[0]) for x in batch]
        labels = torch.tensor([x[1] for x in batch], dtype=torch.int8)

        assert type(documents) == list, 'Needs to be a list of strings'
        tokenized = self.tokenizer(documents, return_tensors='pt', padding=True, truncation=True, max_length=self.max_tokens)

        return tokenized['input_ids'], tokenized['attention_mask'], labels


# from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)


class JEFDataset(Dataset):
    def __init__(self, path, dep_var, text_col, lowercase):
        super().__init__()
        self.dep_var = dep_var
        self.text_col = text_col
        all_columns = [self.dep_var, self.text_col] + ['date_appeal_panel_ruling']

        data = pd.read_parquet(path, columns=all_columns)    

        if len(data) > 600_000:
            print(f'Previous size of training data: {len(data)}. Selecting only last 5 years of the training dataset')
            data.date_appeal_panel_ruling = pd.to_datetime(data.date_appeal_panel_ruling, infer_datetime_format=True, yearfirst=True, dayfirst=False)
            thresh = data.date_appeal_panel_ruling.max() - timedelta(days=365*5)
            data = data[data.date_appeal_panel_ruling >= thresh].copy()
            print(f'New size of training data: {len(data)}')

        data.drop('date_appeal_panel_ruling', axis=1, inplace=True)
        data[self.dep_var] = data[self.dep_var].replace('PROVIMENTO PARCIAL', 'PROVIMENTO')
        data = data[data[self.dep_var].isin(['PROVIMENTO', 'NÃO PROVIMENTO'])]
        data[self.dep_var] = data[self.dep_var].map({'NÃO PROVIMENTO': 0, 'PROVIMENTO': 1})

        if lowercase:
            data[self.text_col] = data[self.text_col].str.lower()

        print(f'Size before: {len(data)} - {path.split("/")[-1]}')        
        data.dropna(inplace=True)
        print(f'Size after: {len(data)} - {path.split("/")[-1]}')
        data.reset_index(drop=True, inplace=True)
        self.data = data.copy()

    def __getitem__(self, idx):
        return self.data.loc[idx, self.text_col], self.data.loc[idx, self.dep_var]

    def __len__(self):
        return len(self.data)

class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.num_labels = len(self.hparams['labels'])
        self.tokenizer = BigBirdTokenizer.from_pretrained(self.hparams['bert_model_path'])
        config = AutoConfig.from_pretrained(self.hparams['bert_model_path'])
        config.__setattr__('num_labels', len(self.hparams['labels']))

        self.accuracy = torchmetrics.Accuracy()
        self.mcc = MCC()
        self.valor_mcc = {'val_mcc': -1, 'test_mcc': -1}
        self.best_mcc = -1.0
        self.precision_metric = torchmetrics.Precision(num_classes=self.num_labels)
        self.recall_metric = torchmetrics.Recall(num_classes=self.num_labels)
        #self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.num_labels)
        self.f1_score = torchmetrics.F1(num_classes=self.num_labels)

        self.model = ModelBigBird(self.hparams['bert_model_path'],
                                      bertconfig=config,
                                      drop_mult=self.hparams.drop_mult,
                                      use_special_classifier=self.hparams.use_special_classifier)

        if self.hparams.bert_unfreeze_mode == 'encoder_last':
            self.model.freeze_bert_encoder()
            self.model.unfreeze_bert_encoder_last_layers()
        elif self.hparams.bert_unfreeze_mode == 'pooler_last':
            self.model.freeze_bert_encoder()
            self.model.unfreeze_bert_encoder_pooler_layer()
        elif self.hparams.bert_unfreeze_mode == 'all':
            self.model.unfreeze_bert_encoder()
        elif self.hparams.bert_unfreeze_mode == 'none':
            self.model.freeze_bert_encoder()

        if self.hparams.weighted_loss:
            weights = torch.FloatTensor(self.hparams.train_weights)
            print(f'Using weighted loss: {weights}')
        else:
            weights = None
        # weight = torch.FloatTensor(self.hparams.train_weights).to(self.hparams.device) if self.hparams.train_weights is not None else None)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.lr = self.hparams.lr
        self.save_hyperparameters()

    def step(self, batch, step_name='train'):
        thresh = self.hparams.thresh_step
        input_ids, attention_masks, y = batch
        logits = self.forward(input_ids, attention_masks).squeeze()
        y = y.type_as(logits)
        loss = self.loss(logits, y)

        if step_name == 'train':
            self.log('train_loss', loss, on_step=True, on_epoch=True,
                     logger=True, prog_bar=True, sync_dist=True, sync_dist_op='mean')
            result = {'train_loss': loss}
            return loss

        else:
            self.log(f'{step_name}_loss', loss, on_step=False, on_epoch=True, logger=True,
                     prog_bar=True, reduce_fx=torch.mean, sync_dist=True, sync_dist_op='mean')

            y_pred = torch.sigmoid(logits)
            y_pred = torch.where(y_pred > thresh, 1.0, 0.0).long()
            y =  y.long()
            #y_pred = torch.argmax(logits, dim=1)
            self.accuracy(y_pred, y)
            #self.mymetric(y_pred, y)
            self.precision_metric(y_pred, y)
            self.recall_metric(y_pred, y)
            self.f1_score(y_pred, y)
            #self.confmat(y_pred, y)
            self.mcc(y_pred, y)

    
        result = {f'{step_name}_loss': loss}
        return result
        
    def calculate_metrics(self, outputs, step_name='val'):
        
        mcc, preds, target = self.mcc.compute()
        tn = ((target == preds) & (target == 0)).sum()
        tp = ((target == preds) & (target == 1)).sum()
        fn = ((target != preds) & (target == 1)).sum()
        fp = ((target != preds) & (target == 0)).sum()

        
        outs = {}
        outs[f'{step_name}_acc'] = self.accuracy.compute()
        outs[f'{step_name}_loss'] = torch.mean(torch.tensor([i[f'{step_name}_loss'] for i in outputs]))
        outs[f'{step_name}_tn'] = tn
        outs[f'{step_name}_fp'] = fp
        outs[f'{step_name}_fn'] = fn
        outs[f'{step_name}_tp'] = tp
        outs[f'{step_name}_f1_score'] = self.f1_score.compute()
        outs[f'{step_name}_precision'] = self.precision_metric.compute()
        outs[f'{step_name}_recall'] = self.recall_metric.compute()
        outs[f'{step_name}_mcc'] = mcc
        #outs[f'{step_name}_mccold'] = mcc2
        #confmat = self.confmat.compute().long().detach().cpu().numpy()

        self.recall_metric.reset()
        self.precision_metric.reset()
        self.f1_score.reset()
        self.accuracy.reset()
        self.mcc.reset()
        #self.confmat.reset()

        if float(mcc) > self.best_mcc:
            self.best_mcc = float(mcc)
            self.log('best_mcc', mcc)
        if self.valor_mcc[f'{step_name}_mcc'] < float(outs[f'{step_name}_mcc']):
            self.valor_mcc[f'{step_name}_mcc'] = float(outs[f'{step_name}_mcc'])

        print(matthews_corrcoef(preds, target), mcc, len(target), len(preds))
        if self.global_rank == 0:
            print(matthews_corrcoef(preds, target), mcc, len(target), len(preds))
            if self.valor_mcc[f'{step_name}_mcc'] < float(outs[f'{step_name}_mcc']):
                self.valor_mcc[f'{step_name}_mcc'] = float(outs[f'{step_name}_mcc'])
                #self.logger.experiment.log({f"best_mcc-confusion_matrix" : wandb.plot.confusion_matrix(preds=preds, y_true=target, class_names=[i[:3].upper() for i in self.hparams.labels])}, commit=False)
            #print("\n\nCONFUSION MATRIX:\n", confmat, "\n")
            print(f"{step_name}_acc: {float(outs[f'{step_name}_acc']):.5f}")
            print(f"{step_name}_mcc: {float(outs[f'{step_name}_mcc']):.5f}")
            print(f"Number of cases: {int(tn+fp+fn+tp)}")
            print('\n')


        for k, v in outs.items():
            self.log(k, v)

    def forward(self, input_ids, attention_masks, *args):
        return self.model(input_ids, attention_masks, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def validation_epoch_end(self, outputs: List[dict]):
        return self.calculate_metrics(outputs, step_name='val')

    def test_epoch_end(self, outputs: List[dict]):
        return self.calculate_metrics(outputs, step_name='test')

    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        #print(self.hparams.cat_names)
        return DataLoader(
            JEFDataset(ds_path, self.hparams.dep_var, self.hparams.text_col, self.hparams.lowercase), 
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
            num_workers=int(cpu_count()),
            collate_fn=EncodeCollateFn(self.tokenizer)
        )

    def setup(self, stage):
        if stage == 'fit':
            total_devices = max(1, self.hparams.n_gpus) * 1
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = math.ceil((self.hparams.epochs * train_batches) / self.hparams.accumulate_grad_batches)
            self.train_steps = int(math.ceil(self.train_steps * 1.01))#1.04)

    def configure_optimizers(self):
        train_steps = self.train_steps
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=self.lr,
                                                           total_steps=train_steps,
                                                           three_phase=True,
                                                           epochs=self.hparams.epochs)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--drop_mult", type=float)
    parser.add_argument("--use_special_classifier", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--thresh_step", type=float)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--gradient_clip_val", type=float)
    parser.add_argument("--lowercase", type=str),
    parser.add_argument("--stochastic_weight_avg", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.valid_path
    test_path = args.test_path

    assert os.path.exists(train_path), f"File doesn't exist: {train_path}"
    assert os.path.exists(val_path), f"File doesn't exist: {val_path}"
    assert os.path.exists(test_path), f"File doesn't exist: {val_path}"

    df_train = pd.read_parquet(train_path, columns=['label'])          
    df_train.label = df_train.label.replace('PROVIMENTO PARCIAL', 'PROVIMENTO')
    df_train.label = df_train.label.replace('NÃO PROVIMENTO', 0)
    df_train.label = df_train.label.replace('PROVIMENTO', 1)
    correct_output = torch.tensor(df_train.label.values)
    trn_wgts = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1)
    trn_wgts = trn_wgts.cpu().numpy()

    del(df_train)

    hparams = Namespace(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=args.batch_size, #12 3090
        epochs=args.epochs, #7
        drop_mult=args.drop_mult,
        use_special_classifier=args.use_special_classifier,
        lowercase=str2bool(args.lowercase),
        lr=args.lr,
        thresh_step=args.thresh_step,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        stochastic_weight_avg=str2bool(args.stochastic_weight_avg),
        dep_var = 'label',
        text_col = "preprocessed_full_text_first_instance_court_ruling",
        bert_model_path='./bigbird-jus',
        labels=[0, 1],
        sync_batchnorm=True,
        device='cuda',
        train_weights=trn_wgts,
        bert_unfreeze_mode='encoder_last',  # 'encoder_last', 'pooler_last', 'all', 'none'
        weighted_loss=True,
        precision='bf16' ,
        n_gpus= 2, 
        deterministic=True,
    )
    module = TrainingModule(hparams)
    gc.collect()
    torch.cuda.empty_cache()

    PROJECT_NAME = args.project_name 
    EXPERIMENT_NAME = args.experiment_name

    lr_logger = LearningRateMonitor(logging_interval='step', log_momentum=True)
    
    if len(EXPERIMENT_NAME) > 2:
        wandb_logger = WandbLogger(name=EXPERIMENT_NAME, project=PROJECT_NAME, offline=False)
    else:
        wandb_logger = WandbLogger(project=PROJECT_NAME)
        EXPERIMENT_NAME = wandb_logger.experiment.name

    wandb_logger.log_hyperparams(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath = os.getcwd(),
        filename= f'{EXPERIMENT_NAME}-best',
        save_top_k=1,
        verbose=True,
        monitor='val_mcc',
        mode='max',
    )
    trainer = pl.Trainer(gpus=hparams.n_gpus,
                        gradient_clip_val=hparams.gradient_clip_val,
                        stochastic_weight_avg=hparams.stochastic_weight_avg,
                        max_epochs=hparams.epochs,
                        progress_bar_refresh_rate=1,
                        log_every_n_steps=10,
                        num_sanity_val_steps=2,
                        accumulate_grad_batches=hparams.accumulate_grad_batches,
                        precision=hparams.precision,
                        sync_batchnorm=hparams.sync_batchnorm,
                        accelerator='ddp',
                        callbacks=[lr_logger, checkpoint_callback],
                        logger=wandb_logger,
                        deterministic=hparams.deterministic,
                        )


    trainer.fit(module)
    path = os.path.join(os.getcwd(), f"{EXPERIMENT_NAME}-best.ckpt")
    trainer.test(ckpt_path = path)

if __name__ == "__main__":
    main()