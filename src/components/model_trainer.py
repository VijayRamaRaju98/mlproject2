import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import evalute_models
from src.exception import CustomException



@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainerConfig()

    def initiate_model_trainer(self, train_set, test_set):
        try:

            x_train, y_train, x_test, y_test = train_set[:,:-1], train_set[:-1], test_set[:,:-1], test_set[:,-1]


            models = {
                'tree':DecisionTreeClassifier(),
                'linear':LinearRegression(),
                'gradient':GradientBoostingClassifier()
            }

            model_report:dict = evalute_models(x_train, y_train, x_test, y_test, models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print(best_model_name)
        except Exception as e:
            raise CustomException(e,sys)



