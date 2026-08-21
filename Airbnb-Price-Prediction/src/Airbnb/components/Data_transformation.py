import os 
import sys
import pandas as pd
from src.Airbnb.exception import customeexception
from src.Airbnb.logger import logging
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.Airbnb.ultils.utils import save_object

