import os 
import sys
import pandas as pd
from src.Airbnb.exception import customexception
from src.Airbnb.logger import logging
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.Airbnb.ultils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','Preprocessor.pkl')
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            numerical_cols   = ['amenities','accommodates','bathrooms','latitude','longitude','host_response_rate','number_of_reviews','review_scores_rating','bedrooms','beds']
            categorical_cols = ['property_type','room_type','bed_type','cancellation_policy','cleaning_fee','city','host_identity_verified','instant_bookable','host_has_profile_pic']

            #define which columns should be ordinal-encoded and which should be scaled
            property_type_cat = ['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft', 'Other', 'Guesthouse', 'Bed & Breakfast', 'Bungalow', 'Villa', 'Dorm', 'Guest suite', 'Camper/RV', 'Timeshare', 'Cabin', 'In-law', 'Hostel', 'Boutique hotel', 'Boat', 'Serviced apartment', 'Tent', 'Castle', 'Vacation home', 'Yurt', 'Hut', 'Treehouse', 'Chalet', 'Earth House', 'Tipi', 'Train', 'Cave', 'Casa particular', 'Parking Space', 'Lighthouse', 'Island']
            room_type_cat = ['Entire home/apt', 'Private room', 'Shared room']
            bed_type_cat = ['Real Bed', 'Futon', 'Pull-out Sofa', 'Airbed', 'Couch']
            cancellation_policy_cat = ['strict', 'moderate', 'flexible', 'super_strict_30', 'super_strict_60'],
            cleaning_fee_cat = ['True', 'False']
            city_cat = ['NYC', 'SF', 'DC', 'LA', 'Chicago', 'Boston']
            host_has_profile_pic_cat = ['t', 'f']
            host_identity_verified_cat = ['t', 'f']
            instant_bookable_cat = ['t', 'f']
            logging.info('Pipeline Initiated')
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')), #replace Nan values by median
                ('scaler',StandardScaler())])
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')), #replace Nan values by most_frequent
                ('ordinalencoder',OrdinalEncoder(categories=[property_type_cat, room_type_cat, bed_type_cat, cancellation_policy_cat, cleaning_fee_cat, city_cat, host_has_profile_pic_cat, host_identity_verified_cat, instant_bookable_cat])),# setting ordered features for mapping to ensure squence and fixed.
                ('scaler',StandardScaler())])
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatranssformation")
            raise customexception(e,sys)