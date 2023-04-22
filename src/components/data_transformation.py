import os
import sys
from src.exception import CustomException
from src.logger import logging
#from data_ingestion import DataIngestion
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

TARGET_COLUMN_NAME = "price"

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_config(self):
        """
        This function is responsible for data transformation
        """

        try:

            df = pd.read_csv("artifacts\train.csv")

            df.drop_duplicates(keep = 'first', inplace = True, ignore_index = True)

            categorical_columns = list(df.loc[:,df.dtypes == 'object'].columns)
            
            numerical_columns = list(df.columns)
            for i in categorical_columns:
                if i in numerical_columns:
                    numerical_columns.remove(i)

            num_pipeline = Pipeline(
                steps = [
                ("imputer" , SimpleImputer(strategy = "median")),
                ("scaler" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                ("imputer" , SimpleImputer(strategy= "most_frequent")),
                ("one_hot_encoder" , OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ] 
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

        def initiate_data_transformation(self, train_path, test_path):

            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")

                logging.info("Obtaining preprocessing object")

                categorical_columns = list(train_df.loc[:,train_df.dtypes == 'object'].columns)
            
                numerical_columns = list(train_df.columns)
                for i in categorical_columns:
                    if i in numerical_columns:
                        numerical_columns.remove(i)

                train_df_independent_features = train_df.drop([TARGET_COLUMN_NAME], axis = 1)
                train_df_target_feature = train_df[TARGET_COLUMN_NAME]

                test_df_independent_features = test_df.drop([TARGET_COLUMN_NAME], axis = 1)
                test_df_target_feature = test_df[TARGET_COLUMN_NAME]

                logging.info("Applying the preprocessing object on the training and testing dataframe")

                preprocessing_obj = self.get_data_transformer_config()

                input_feature_train_arr = preprocessing_obj.fit_transform(train_df_independent_features)
                input_feature_test_arr = preprocessing_obj.transform(test_df_independent_features)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(train_df_target_feature)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(test_df_target_feature)
                ]

                logging.info(f"Saved preprocessing object.")

                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                    obj = preprocessing_obj
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )

            except Exception as e:
                raise CustomException(e, sys)

