


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
import keras as K

# local application/library specific imports
from app_config import AppConfig
from app_src.SentenceTransformerEncoderModel import SentenceTransformerEncoderModel

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']
random.seed(RANDOM_STATE)

class CustomEncoder:
    def __init__(self, name, encodder_model, tokenizer_model):
        self.name = name
        self.problem_statement_model = encodder_model
        self.problem_statement_tokenizer = tokenizer_model
        
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
            transformer_outputs = self.problem_statement_model(
                input_ids=encoded_premise.input_ids, 
                attention_mask=encoded_premise.attention_mask
            )

            # Use pooler output for classification
            batch_embeddings = transformer_outputs.last_hidden_state

            # Convert attention mask to float and expand dims to match transformer outputs.
            mask = tf.cast(encoded_premise.attention_mask, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)  # Now shape: (batch_size, sequence_length, 1)

            # Multiply the transformer outputs by the mask to zero-out the padded positions.
            masked_embeddings = batch_embeddings * mask  # shape: (batch_size, sequence_length, hidden_size)

            # Sum the embeddings over the sequence length.
            sum_embeddings = tf.reduce_sum(masked_embeddings, axis=1)  # shape: (batch_size, hidden_size)
            
            # Compute the sum of the mask values to know how many valid tokens there were per example.
            sum_mask = tf.reduce_sum(mask, axis=1)  # shape: (batch_size, 1)
            
            # Avoid division by zero by adding a small epsilon.
            epsilon = tf.keras.backend.epsilon()
            
            # Compute the mean by dividing the summed embeddings by the number of valid tokens.
            mean_embeddings = sum_embeddings / (sum_mask + epsilon)
            
            # Convert to numpy array and append to the list
            all_embeddings.append(mean_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings
    
    def encode_problem_solution(self, solution):
        pass
