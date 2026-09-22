import os
import sys
import pandas as pd
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path= os.path.join("Artifacts","Preprocessor.pkl")
            model_path = os.path.join("Artifacts","Model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled = preprocessor.transform(features)
            pred = model.predict(scaled)
            return pred
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                Airline:str,
                Source:str,
                Destination:str,
                Journey_Day:int,
                Journey_Month: int,
                Dep_Hour: int,
                Dep_Minute: int
                ):
        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Journey_Day = Journey_Day
        self.Journey_Month = Journey_Month
        self.Dep_Hour = Dep_Hour
        self.Dep_Minute = Dep_Minute

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Airline': [self.Airline],
                'Source':[self.Source],
                'Destination':[self.Destination],
                'Journey_Day': [self.Journey_Day],
                'Journey_Month':[self.Journey_Month],
                'Dep_Hour': [self.Dep_Hour],
                'Dep_Minute':[self.Dep_Minute]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)
            
    
    