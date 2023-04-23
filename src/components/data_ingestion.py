import os
import sys
import pandas as pd
import numpy as np
import pyodbc

from exception import CustomException 
from logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig

INPUTFILE = 'Dataset\Clean_Dataset.csv'

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")

class DataValidation:
    try:
        def FileValidation(self, INPUTFILE):
            df = pd.read_csv(INPUTFILE)
            logging.info("Read the input file successfully {}".format(df.columns))

            agreed_columns = ['airline', 'flight', 'source_city', 'departure_time','stops', 
            'arrival_time', 'destination_city', 'class', 'duration', 'days_left', 'price']

            if df.shape[1] == 12:
                df.drop(['Unnamed: 0'], axis = 1, inplace = True)
                logging.info("Dropped the Index column")

            if df.shape[1] == 11:
                for i in df.columns:
                    if i in agreed_columns:
                        logging.info("Columns validated, {} columns are present and names are {}".
                        format(df.shape[1], tuple(df.columns)))
                    else:
                        logging.info("Data is not as per agreed terms, hence rejected")
            else:
                logging.info("Data has {} columns and not as per agreed terms".format(df.shape[1]))  

            #Updating data types

            df['airline'] = df['airline'].astype('object')
            df['flight'] = df['flight'].astype('object')
            df['source_city'] = df['source_city'].astype('object')
            df['departure_time'] = df['departure_time'].astype('object')
            df['stops'] = df['stops'].astype('object')
            df['arrival_time'] = df['arrival_time'].astype('object')
            df['destination_city'] = df['destination_city'].astype('object')
            df['class'] = df['class'].astype('object')

            df['duration'] = df['duration'].astype('float64')

            df['days_left'] = df['days_left'].astype('int64')
            df['price'] = df['price'].astype('int64')

            logging.info("Data types checked for all columns")
        
            return df

    except Exception as e:
        raise CustomException(e, sys)

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self, df):
        logging.info("Entered the data ingestion method or component")
        try:
            conn=pyodbc.connect(Trusted_connection='yes',Driver='{ODBC Driver 17 for SQL Server}',
                                Server='DESKTOP-11KNG3S', Database='Airline_Analysis')

            cursor=conn.cursor()

            cursor.execute("IF NOT EXISTS (SELECT * FROM SYSOBJECTS WHERE name='FLIGHT' and xtype='U') CREATE TABLE FLIGHT (airline NVARCHAR(50), flight NVARCHAR(50), source_city NVARCHAR(50), departure_time NVARCHAR(50),stops NVARCHAR(50), arrival_time NVARCHAR(50), destination_city NVARCHAR(50), class NVARCHAR(50), duration FLOAT,days_left INTEGER, price INTEGER)")

            for index,row in df.iterrows():
                cursor.execute("INSERT INTO Airline_Analysis.dbo.FLIGHT (airline,flight,source_city,departure_time,stops,arrival_time,destination_city,class,duration,days_left,price)"
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    row[10]
                    )
                
            cursor.execute("IF EXISTS (SELECT * FROM SYSOBJECTS WHERE name='FLIGHT_STAGGING' and xtype='U') DROP TABLE Airline_Analysis.dbo.FLIGHT_STAGGING")

            cursor.execute("select distinct airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left, price into Airline_Analysis.dbo.FLIGHT_STAGGING from Airline_Analysis.dbo.FLIGHT")

            cursor.execute("truncate table Airline_Analysis.dbo.FLIGHT")

            cursor.execute("insert into Airline_Analysis.dbo.FLIGHT select * from Airline_Analysis.dbo.FLIGHT_STAGGING")

            conn.commit()

            logging.info("Data inserted in the database successfully")

            logging.info("Fetching complete data from the database")

            cursor.execute('select top 10000* from [dbo].[FLIGHT]')  #AFTER COMPLETION FETCH COMPLETE DATA
            data_fetch = cursor.fetchall()

            df_sql = pd.DataFrame(np.array(data_fetch) , columns = ['airline', 'flight', 'source_city', 'departure_time', 
                                                        'stops', 'arrival_time', 'destination_city', 'class', 
                                                        'duration', 'days_left', 'price'])

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df_sql.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Training and Testing dataset split initiated")

            train_set, test_set = train_test_split(df_sql, test_size = 0.2, random_state = 101)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    cls = DataValidation()
    df = cls.FileValidation(INPUTFILE)
    logging.info("Data validated and read successfully")
    train_data,test_data=obj.initiate_data_ingestion(df)
    logging.info("Data inserted into database successfully")
    



