import os,sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.exception import CustomException

@dataclass
class DatatransformationConfig:
    preprocessor_file_obj = os.path.join('artifacts', 'preprocessor.pkl')

    
class DataTransformation:
    def __init__(self):
        self.dataingestionconfig = DatatransformationConfig()

    def get_preprocessor_obj(self):

        try:
            num_cols = ["writing_score", "reading_score"]
            cat_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('std', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one', OneHotEncoder()),
                ('std', StandardScaler(with_mean=False))
            ])

            preprocessor_obj = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])
            
            return preprocessor_obj

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_column = 'math_score'
            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            preprocessor = self.get_preprocessor_obj()

            input_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_test_arr = preprocessor.transform(input_feature_test_df)

            train = np.c_[input_train_arr, target_feature_train_df]
            test = np.c_[input_test_arr, target_feature_test_df]

            return train, test
            
        except Exception as e:

            raise CustomException(e,sys)




