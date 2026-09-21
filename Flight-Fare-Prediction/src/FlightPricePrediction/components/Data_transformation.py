import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OrdinalEncoder
from src.FlightPricePrediction.utils.utils import save_object
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder #Onehot Encoding

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','Preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')
            #Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute']
            categorical_cols = ['Airline', 'Source', 'Destination']
            airline_categories = ["Jet Airways", "IndiGo", "Air India","Multiple carriers","SpiceJet","Vistara","Air Asia","GoAir","Multiple carriers Premium economy","Jet Airways Business","Vistara Premium economy","Trujet"]
            source_categories = ["Delhi","Kolkata","Banglore","Mumbai","Chennai"]
            destination_categories = ["Cochin","Banglore","Delhi","New Delhi","Hyderabad","Kolkata"]
            logging.info("Pipeline Initiated")
            
            #numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                    ]
            )
            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    ))
                ]
            )
            preprocessor = ColumnTransformer(
                    [
                        ('num_pipeline',num_pipeline, numerical_cols),
                        ('cat_pipeline',cat_pipeline,categorical_cols)
                    ]
                )
            return preprocessor
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            target_column_name='Price'
            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df  = test_df[target_column_name]
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]
            
            logging.info(f'Train Array : \n{train_arr}')
            logging.info(f'Test Array : \n{test_arr}')
            #Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("Preprocessing joblib file saved")
            
            return (train_arr,test_arr)
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)  