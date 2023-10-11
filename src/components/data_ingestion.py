import pandas as pd
import numpy as np
from dataclasses import dataclass
import os,sys
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        

        df = pd.read_csv("../project/notebook/data/stud.csv")
        os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path), exist_ok=True)
        df.to_csv(self.dataingestionconfig.raw_data_path, index=False, header=True)
        train_set, test_set = train_test_split(df, test_size=0.2)
        train_set.to_csv(self.dataingestionconfig.train_data_path, index=False, header=True)
        test_set.to_csv(self.dataingestionconfig.test_data_path, index=False, header=True)
        

        return self.dataingestionconfig.train_data_path, self.dataingestionconfig.test_data_path 
    


if __name__=="__main__":
    obj = DataIngestion()
    #print(obj.dataingestionconfig.train_data_path, obj.dataingestionconfig.test_data_path)
    train_path, test_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformer(train_path, test_path)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    #score = model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
   
