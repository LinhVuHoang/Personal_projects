import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception
from src.Airbnb.ultils.utils import save_object
from src.Airbnb.ultils.utils import evaluate_model

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifacts','Model.pkl')
    
    
class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
  
            X_train, y_train, X_test, y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )
            models = {
                'XGBoost': XGBRegressor(n_estimators=800,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    gamma=0.1,
                    min_child_weight=3,
                    random_state=42,
                    n_jobs=-1),
                'Catboost': CatBoostRegressor(iterations=800,
                    learning_rate=0.1,
                    depth=8,
                    subsample=1.0,
                    l2_leaf_reg=3,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=42),
                'Lightgmb': LGBMRegressor(n_estimators=800,
                    min_child_samples=20,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=50,
                    subsample=0.9,
                    colsample_bytree=0.7,
                    random_state=42,
                    n_jobs=-1)
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            #to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
        except Exception as e:
            logging.info('Exception occured at Model training')
            raise customexception(e,sys)