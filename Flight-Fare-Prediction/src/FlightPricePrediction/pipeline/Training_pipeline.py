#Data ingestion pipeline

from src.FlightPricePrediction.components.Data_ingestion import DataIngestion
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

#Data Transformation Pipeline

from src.FlightPricePrediction.components.Data_transformation import DataTransformation
data_transformation = DataTransformation()

train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path,test_data_path)

#Data training pipeline

from src.FlightPricePrediction.components.Model_trainer import ModelTrainer

model_trainer_obj = ModelTrainer()

model_trainer_obj.initate_model_training(train_arr,test_arr)

#Data evaluate pipeline

from src.FlightPricePrediction.components.Model_evaluation import ModelEvaluation

model_evaluate_obj = ModelEvaluation()

model_evaluate_obj.initiate_model_evaluation(train_arr,test_arr)

