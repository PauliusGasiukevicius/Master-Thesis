# About
Source code and datasets of my master thesis "Clustering and Visualization of Phishing emails using Self-Organizing Maps"

# Datasets
3 datasets from public domains are included:
-	Phishing corpus phishing email dataset (https://academictorrents.com/details/a77cda9a9d89a60dbdfbe581adf6e2df9197995a)
-	Kaggle SPAM and HAM dataset (https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset)
-	SPAM Archive 2021 Jan-Feb dataset (http://untroubled.org/spam/)

# Code
- `parse_dataset.py` - has some utility function for parsing email datasets either from raw emails in txt/eml or from csv files, combines above mentioned datasets and processes them into word root frequency matrix for SOM training
- `SOM_with_lib.py` - uses minisom and matplotlib to cluster datasets and save resulting vizualization images

# Clustering Sample results
Some of clustering visualization results are provided in `/results` directory
all files follow the following format: `A_sz_B_fr_C_w_D_sz_E_e_F_lr_G_H_I_J_sigma_K` where meaning of capital letter variables like `A` is given in table below

| Variable  | Meaning |
| ------------- | ------------- |
| A  | limit of emails of each class (phishing/spam/ham) in dataset slice  |
| B  | frequency cut-off - how frequent some word root must be in whole dataset to be included in attributes |
| C  | minimum amount of frequent words that must be present in each dataset sample to be included in training samples (helps to filter non-english emails) |
| D  | distance calculation used to train SOM (euclidean/manhattan) |
| E  | length of SOM output neurons matrix side |
| F  | number of epochs network was trained on  |
| G  | learning rate used during SOM training  |
| H  | neighborhood function used to train SOM (gaussian/bubble/mexical hat)  |
| I  | topology of SOM output neurons (rectangular/hexagonal)  |
| J  | activation function used to train SOM (euclidean/manhattan) |
| K  | neighborhood radius (a.k.a. sigma)  |

# Licence
Licensed under the MIT License, Copyright ©Paulius Gasiukevičius (2021-present)
