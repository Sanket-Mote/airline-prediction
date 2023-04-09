import os
import sys
import pandas as pd
import pyodbc

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")

class DataIngestion:
    
