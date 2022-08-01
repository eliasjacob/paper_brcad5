<div align="center">    
 
# Using deep learning to predict outcomes of legal appeals better than human experts: A study with data from Brazilian federal courts   

[![Paper](https://zenodo.org/badge/doi/10.1371/journal.pone.0272287.svg)](https://doi.org/10.1371/journal.pone.0272287)
[![Datasets](https://img.shields.io/badge/dataset-BrCAD5-red)](https://www.kaggle.com/datasets/eliasjacob/brcad5)


</div>
 
## Description   
Legal scholars have been trying to predict the outcomes of trials for a long time. In recent years, researchers have been harnessing advancements in machine learning to predict the behavior of natural and social processes. At the same time, the Brazilian judiciary faces a challenging number of new cases every year, which generates the need to improve the throughput of the justice system. Based on those premises, we trained three deep learning architectures, ULMFiT, BERT, and Big Bird, on 612,961 Federal Small Claims Courts appeals within the Brazilian 5th Regional Federal Court to predict their outcomes. We compare the predictive performance of the models to the predictions of 22 highly skilled experts. All models outperform human experts, with the best one achieving a Matthews Correlation Coefficient of 0.3688 compared to 0.1253 from the human experts. Our results demonstrate that natural language processing and machine learning techniques provide a promising approach for predicting legal outcomes. We also release the Brazilian Courts Appeal Dataset for the 5th Regional Federal Court (BrCAD-5), containing data from 765,602 appeals to promote further developments in this area.


## How to run   
1 - Clone this project

2 - Download all the parquet and csv files from [here](https://www.kaggle.com/eliasjacob/brcad5) and place all files inside `data/`

3 - Use the same kaggle page described above to download the pretrained language models and place all files and folders inside the folder `code/` in this repository. Replace any existing files.

4 - Install the environment
```bash
cd paper_brcad5
conda env create --file code/environment.yaml
conda activate paper_brcad5
```

5 - See files with the name starting with numbers `01` to `07` inside the folder `code/` to reproduce our results 

### Citation   
```
@article{menezes-neto+clementino,
  title={Using deep learning to predict outcomes of legal appeals better than human experts}: A study with data from Brazilian federal courts,
  author=Elias Jacob de {Menezes-Neto} and Marco Bruno Miranda {Clementino},
  journal={Location},
  year={2022},
  
}
```   
