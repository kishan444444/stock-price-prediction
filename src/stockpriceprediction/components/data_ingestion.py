import numpy as np
import pandas as pd
from src.stockpriceprediction.logger import logging
from src.stockpriceprediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import os
import sys

class Dataingestionconfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    
class Dataingestion:
    def __init__(self):
        self.ingestion_config=Dataingestionconfig()
     
    def initiate_data_ingestion(self):
        logging.info("data_ingestion_started")
        
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","stock prices data.csv")))
            logging.info("I have read the dataset as df")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("i have saved the raw dataset in artifacts folder")
            
            logging.info("here i had perform train_test_split")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train_test_split completed")
            
            
            data.to_csv(self.ingestion_config.train_data_path,index=False)
            data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("data ingestion part completed")
            
            return (
                 
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
            logging.info("exception occured at data ingestion stage")
            raise customexception(e,sys)
           
