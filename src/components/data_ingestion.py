import os
import sys
sys.path.insert(0, './src')
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")
    
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try: # reading data here from csv, can read from other sources as well here only like sql, mongo etc..
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Read the dataset as a dataframe.")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
    
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train Test split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("ingestion of data is completed.")
            
            return (
                self.ingestion_config.train_data_path, # this info would be useful for data transformation ahead
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()