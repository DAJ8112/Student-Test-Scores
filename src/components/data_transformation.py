import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from skelearn.impute import SimpleImputer
from sklearn.pipeline import pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "./src")

from exception import CustomException
from logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        pass
