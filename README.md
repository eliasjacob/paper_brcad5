<div align="center">    
 
# Using deep learning to predict outcomes of legal appeals better than human experts     

[![Paper](http://img.shields.io/badge/paper-XXXXX.11111.2222-3333.svg)](https://github.com/eliasjacob/paper_brcad5)


</div>
 
## Description   
Legal scholars have been trying to predict the outcomes of trials for a long time. In recent years, researchers have been harnessing advancements in machine learning to predict the behavior of natural and social processes. At the same time, the Brazilian judiciary faces a challenging number of new cases every year, which generates the need to better understand all steps of a typical case flow through the justice system. Based on those premises, we trained three deep learning architectures, ULMFiT, BERT, and Big Bird, on 612,961 Federal Small Claims Courts appeals within the Brazilian 5th Regional Federal Court to predict their outcomes. We compare the performance of the machines with 22 highly skilled experts in their ability to foresee the results of appeals. All models outperform human experts, with the best one achieving a Matthews Correlation Coefficient of 0.3688 against 0.1253 from the humans. Our results show the real-world usefulness of machine learning within the legal field by demonstrating unprecedented success when compared with analysis by human experts. We also release the Brazilian Courts Appeal Dataset for the 5th Regional Federal Court (BrCAD-5), containing data from 765,602 appeals to promote further developments in this area.

## How to run   
1 - Clone this project

2 - Download the data from [here](https://www.kaggle.com/eliasjacob/brcad5) and place all files under `data/`

3 - Download the pretrained language models from [here](https://jacob.al/paper_brcad5) and place all files and folders under this repository folder.

4 - Install the enviroment
```bash
cd paper_brcad5
conda env create --file environment.yaml
conda activate paper_brcad5
```

5 - See files with the name beggiging with numbers `01` to `07` to reproduce our results 

### Citation   
```
@article{menezes-neto+clementino,
  title={Using deep learning to predict outcomes of legal appeals better than human experts},
  author=Elias Jacob de {Menezes-Neto} and Marco Bruno Miranda {Clementino},
  journal={Location},
  year={2022},
  
}
```   
