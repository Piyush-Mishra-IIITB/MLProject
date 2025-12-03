import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            project_root = Path(__file__).resolve().parents[2]  # go up to MLPROJECT folder
            file_path = project_root / "notebook" / "data" / "stud.csv"

            df = pd.read_csv(file_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # --- CLEAN UP NaNs and infs before training ---
    import numpy as np

    print("NaNs in train array:", np.isnan(train_arr).sum())
    print("NaNs in test array:", np.isnan(test_arr).sum())
    print("Infs in train array:", np.isinf(train_arr).sum())
    print("Infs in test array:", np.isinf(test_arr).sum())

    train_arr = np.nan_to_num(train_arr, nan=0.0, posinf=0.0, neginf=0.0)
    test_arr = np.nan_to_num(test_arr, nan=0.0, posinf=0.0, neginf=0.0)

    print("After cleanup:")
    print("NaNs in train array:", np.isnan(train_arr).sum())
    print("NaNs in test array:", np.isnan(test_arr).sum())
    print("Infs in train array:", np.isinf(train_arr).sum())
    print("Infs in test array:", np.isinf(test_arr).sum())
    # ------------------------------------------

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))


