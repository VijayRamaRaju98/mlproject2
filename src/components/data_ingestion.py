import pandas as pd
import numpy as np
from dataclasses import dataclass
import os,sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException




@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:

            df = pd.read_csv("../project/notebook/data/stud.csv")
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path), exist_ok=True)
            df.to_csv(self.dataingestionconfig.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2)
            train_set.to_csv(self.dataingestionconfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.dataingestionconfig.test_data_path, index=False, header=True)
        

            return self.dataingestionconfig.train_data_path, self.dataingestionconfig.test_data_path 
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    #print(obj.dataingestionconfig.train_data_path, obj.dataingestionconfig.test_data_path)
    train_path, test_path = obj.initiate_data_ingestion()