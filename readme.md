# FISE3 - Project BigData NoSQL
<img src="https://www.telecom-st-etienne.fr/wp-content/uploads/sites/3/2015/12/telecom_saint_etienne_logo_transparent-768x943.png" width="96" height="117">
<img src="https://www.bgp4.com/wp-content/uploads/2019/08/Scikit_learn_logo_small.svg_-840x452.png" width="105" height="56">


## Aim
This project concerns the analysis and exploitation of listings of rental properties on Airbnb in the city of Bordeaux, , as well as the prediction of the price of a night.

![Airbnb_Bordeaux](bordeaux_airbnb.png)

The students that worked on this project : **Mohammed BENMOUSSA, Youssef BICHICHI, Mouad RIALI, Jinda WU.**

We have used Python, Jupyter Notebook to explore the data and generate a good model for price prediction.

## Getting Started 
To run this repo, be sure to install the following environment and library:
- Python
- Pandas
- Numpy
- sklearn
- matplotlib
- seaborn
- scipy
- statsmodels
- nltk
- folium
- re
- xgboost

## Dataset
Bordeaux airbnb dataset you can find it in this repo.

## Run
To run iPython file, you need to run jupyter notebook. Type this in CMD.
```
jupyter notebook
```

## Cloud Setup Instructions
Download the Dataset needed for running the code from [here](https://mootse.telecom-st-etienne.fr/mod/resource/view.php?id=29857).

### Instructions to setup the cloud services :
```
print("copier le fichier source csv de la machine physique vers la machine virtuelle qui contient cluster hadoop par scp (secure copy)")
os.system('scp -P '+PORT_HADOOP+' '+ SOURCE_FILE + ' root@127.0.0.1:/root;')
print("mettre le fichier csv sur hdfs par ssh")
os.system('ssh -t root@127.0.0.1 -p '+PORT_HADOOP+' "hdfs dfs -put /root/'+SOURCE_FILE_NAME+' /input;"')
print("recuperer le fichier csv du hdfs sur la machine virtuelle hadoop")
os.system('ssh -t root@127.0.0.1 -p '+PORT_HADOOP+' "hdfs dfs -get /input/'+SOURCE_FILE_NAME+' /root/middleRep;"')
print("copier le fichier de la machine virutelle hadoop vers la machine physique qui est windows dans notre cas")
os.system('scp -P '+PORT_HADOOP+' root@127.0.0.1:/root/middleRep/'+SOURCE_FILE_NAME+' '+LOCAL_FOLDER+'')
print("faire une configuration aws")
os.system('aws configure')
print("copier le fichier de la machine physique vers l'instance EC2 ")
os.system('scp -i ./'+KEY_FILE_NAME+' .\\'+SOURCE_FILE_NAME+' ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\preprocess.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\predict.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\utils.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
#os.system('scp -i ./'+KEY_FILE_NAME+' .\\requirements.txt ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
print("copier le fichier de l'instance EC2 vers la machine physique ")
os.system('scp -i ./'+KEY_FILE_NAME+' ec2-user@'+PUBLIC_IPV4_DNS+':/home/predict.json '+LOCAL_FOLDER)
print("se connecter en ssh sur l'instance EC2")
os.system('ssh -i "'+KEY_FILE_NAME+'" ec2-user@'+PUBLIC_IPV4_DNS+'')
#python3 -m venv ./ml
#source ./ml/bin/activate
#pip3 install -r requirements.txt
#python preprocess.py
#python predict.py
```
### Instructions to setup the predicted value into data collection in mongodb :
```
import json
from pymongo import MongoClient 

myclient = MongoClient("mongodb://localhost:27017/") 
# database 
db = myclient["airbnb"]
# created or Switched to collection
Collection = db["data"]
# Loading or Opening the json file
with open('predict.json') as file:
    file_data = json.load(file)
# inserting the loaded data in the Collection
if isinstance(file_data, list):
    Collection.insert_many(file_data)  
else:
    Collection.insert_one(file_data)
```

### visualization
```
!pip3 install pymongo
import pymongo
from pymongo import MongoClient
import folium
from folium.plugins import FastMarkerCluster

client=MongoClient()
db=client["airbnb"]#name of database
collection=db["data"]#name of collection
map1 = folium.Map(location=[44.8350, -0.5800], zoom_start=12.5)

for x in collection.find():
    message="prix = "+str(x["prix"])+"€"
    folium.Marker([x["longitude"], x["laltitude"]], popup=message).add_to(map1)
map1
```
## Methodology
1. Importing the Essential Libraries, Metrics
2. Exploratory Data Analysis
3. Feature Selection & Checking for the missing values
4. Data Visualization
5. Standardizing the Data
6. Analysis of various machine learning models.
7. Fine-tuning the models
8. Data transportation from a virtual machine to the cloud services.
9. Conclusions

## Results
You will find results printed in the notebook.
Either run the notebook or you can find the pdf format of the notebook in this repo.

## Useful links & References
- [House-price-prediction](https://www.kaggle.com/emrearslan123/house-price-prediction) 
- https://scikit-learn.org/stable/
