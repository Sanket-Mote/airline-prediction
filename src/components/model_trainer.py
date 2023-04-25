import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, .
    r2_score
)

from logger import logging
from exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        try:
            self.model_trainer_config=ModelTrainerConfig()

        except Exception as e:
            raise CustomException(e, sys)

        def initiate_model_trainer(self, train_arr, test_arr):
            try:
                logging.info("Splitting the training and testing input data")

                X_train, y_train, X_test, y_test = (
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:,:-1],
                    test_arr[:,-1]
                )

                models = {
                    "Random Forest" : RandomForestRegressor(),
                    "Decision Tree" : DecisionTreeRegressor(),
                    "Gradient Boosting" : GradientBoostingRegressor(),
                    "Linear Regression" : LinearRegression(),
                    "XGBRegressor" : XGBRegressor(),
                    "Catboost Regressor" : CatBoostRegressor(verbose = False),
                    "Adaboost Regressor" : AdaBoostRegressor()
                }

                params = {

                }
                

            except Exception as e:
                raise CustomException(e, sys)
