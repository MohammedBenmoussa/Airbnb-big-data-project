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