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

# Licence
Licensed under the MIT License, Copyright ©Paulius Gasiukevičius (2021-present)
