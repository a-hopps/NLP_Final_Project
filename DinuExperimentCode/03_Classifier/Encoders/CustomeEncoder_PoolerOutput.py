


# standard library imports
import os

# set os environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random

# related third-party
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from math import ceil

# local application/library specific imports
from app_config import AppConfig

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']
random.seed(RANDOM_STATE)

class CustomEncoder:
    def __init__(self, encodder_model):
        self.problem_statement_tokenizer = AutoTokenizer.from_pretrained(encodder_model)
        self.problem_statement_model = TFAutoModel.from_pretrained(encodder_model)
        
    def encode_problem_statement(self, problem_statements, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices(problem_statements).batch(batch_size)
        all_embeddings = []

        # Check if GPU is available
        if tf.config.experimental.list_physical_devices('GPU'):
            print("Using GPU")
        else:
            print("GPU not available, using CPU")

        # Calculate total number of batches
        total_batches = ceil(len(problem_statements) / batch_size)
        
        for batch in tqdm(dataset, total=total_batches, desc="Encoding problem statements"):

            # Convert each element in the batch (a tf.Tensor) to a Python string.
            # If your strings are stored as bytes, decode them with 'utf-8'.
            batch_strings = [s.numpy().decode('utf-8') for s in batch]
            encoded_premise = self.problem_statement_tokenizer(
                batch_strings,                                        # Encode each sentence in the batch
                add_special_tokens=True,                            # Compute the CLS token
                truncation=True,                                    # Truncate the embeddings to max_length
                max_length=512,                                     # Pad & truncate all sentences.
                padding='max_length',                               # Pad the embeddings to max_length
                return_attention_mask=True,                         # Construct attention masks.
                return_tensors='tf'                                 # Return TensorFlow tensors.
            )

            # Ensure the operations run on the GPU
            embeddings = self.problem_statement_model(
                input_ids=encoded_premise.input_ids, 
                attention_mask=encoded_premise.attention_mask
            )

            # Use pooler output for classification
            batch_embeddings = embeddings.pooler_output

            # Convert to numpy array and append to the list
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings
    
    def encode_problem_solution(self, solution):
        pass
    


