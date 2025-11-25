import os
import pandas as pd
import numpy as np
import ast
import random

import tensorflow as tf
from transformers import AutoTokenizer

import keras as K


# local application/library specific imports
from app_src.MultiHeadEditorialTransformerWrapper import MultiHeadEditorialTransformerWrapper
from app_config import AppConfig
from app_src.common import set_random_seed

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class MultiHeadEditorialTransformerEvaluator():
    
    def __init__(self) -> None:        
        self.encoder_collection = []
        
        set_random_seed(RANDOM_STATE)
        
        self.__define_models()

    def __define_models(self):

        self.encoder_collection = [
            'sentence-transformers/all-mpnet-base-v2',
            # 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            # 'sentence-transformers/multi-qa-distilbert-cos-v1',
            # 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            # 'sentence-transformers/all-distilroberta-v1',
            # 'sentence-transformers/all-MiniLM-L12-v2',
            # 'sentence-transformers/all-MiniLM-L6-v2',
            # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            # 'sentence-transformers/paraphrase-albert-small-v2',
            # 'microsoft/codebert-base',
            # 'roberta-base',
            # 'bert-base-uncased'
        ]
    
    def __save_metrics(self, encoder, estimator_name, metrics_results, number_of_tags):
        
        scores = {key: str(value) for key, value in metrics_results.items()}
        csv_headers = ['Encoder Name', 'Estimator Name'] + list(scores.keys())
        output_data = [f'{encoder}', f'{estimator_name}'] + list(scores.values())

        # Create a DataFrame
        df = pd.DataFrame([output_data], columns=csv_headers)

        if not os.path.isfile(CONFIG[f'TOP_{number_of_tags}_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH'], index=False, mode='a', header=False)

    def evaluate_model(self, epochs, batch_size, number_of_tags=5, model=None, transformer_model_path=None, train_dataset_path=None, val_dataset_path=None, test_dataset_path=None):
        if model is None:
            raise Exception("Model name is required for evaluation")
        else:
            print(f"Training and evaluating model: {model}")
            transformer_wrapper = MultiHeadEditorialTransformerWrapper(model, number_of_tags)
            transformer_wrapper.train_model(
                train_dataset_path=train_dataset_path,
                val_dataset_path=val_dataset_path,
                epochs=epochs,
                batch_size=batch_size,
                transformer_model_path=transformer_model_path
            )
            
            metrics = transformer_wrapper.benchmark_model(
                test_dataset_path=test_dataset_path,
                batch_size=32
            )
            
            self.__save_metrics(model, model, metrics, number_of_tags)


    def evaluate_models(self, epochs, batch_size, number_of_tags=5, transformer_model_path=None, train_dataset_path=None, val_dataset_path=None, test_dataset_path=None):
        for model in self.encoder_collection:
            self.evaluate_model(epochs, batch_size, number_of_tags, model, transformer_model_path, train_dataset_path, val_dataset_path, test_dataset_path)

