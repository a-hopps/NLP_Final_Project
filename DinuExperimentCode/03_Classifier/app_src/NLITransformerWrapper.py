import pandas as pd
import random
import tensorflow as tf


# local application/library specific imports
from app_src.NLITransformerEmbeddingModel import NLITransformerEmbeddingModel
from app_src.NLITripletTrainer import NLITripletTrainer
from app_config import AppConfig
from app_src.common import set_random_seed
from transformers import TFAutoModel, AutoTokenizer
from keras.optimizers import AdamW
import numpy as np

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class NLITransformerWrapper():
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.train_dataset = None
        
        set_random_seed(RANDOM_STATE)
    
    def __tokenize_data(self, data):

        tokens = self.tokenizer(
            data,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf'
        )
        return tokens['input_ids'], tokens['attention_mask']    
    
    def __build_tf_dataset(self, train_dataset_path, batch_size = 32, shuffle_buffer_size=10000):
        self.train_dataset = pd.read_csv(train_dataset_path)

        problem_statements = self.train_dataset['problem_statement'].tolist()
        entailments = self.train_dataset['entailment'].tolist()
        contradictions = self.train_dataset['contradiction'].tolist()
        
        problem_statements_input_ids, problem_statements_attention_mask = self.__tokenize_data(problem_statements)
        entailments_input_ids, entailments_attention_mask = self.__tokenize_data(entailments)
        contradictions_input_ids, contradictions_attention_mask = self.__tokenize_data(contradictions)
        
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': problem_statements_input_ids,
                'attention_mask': problem_statements_attention_mask
            },
            {
                'input_ids': entailments_input_ids,
                'attention_mask': entailments_attention_mask
            },
            {
                'input_ids': contradictions_input_ids,
                'attention_mask': contradictions_attention_mask
            }
        ))
        
        tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)
        tf_dataset = tf_dataset.cache()  # Use caching if your dataset fits in memory; otherwise, consider file-based caching.
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset
    
    
    def train_model(self, train_dataset_path, epochs=5, batch_size=32, transformer_model_path=None):
        if transformer_model_path:
            self.transformer_model = TFAutoModel.from_pretrained(transformer_model_path)
        else:
            self.transformer_model = TFAutoModel.from_pretrained(self.model_name)
        
        self.encoder_model = NLITransformerEmbeddingModel(self.transformer_model)
        
        # Unfreeze the transformer layers
        self.encoder_model.unfreeze_transformer()
        
        self.triplet_trainer = NLITripletTrainer(self.encoder_model)
        
        self.triplet_trainer.compile(optimizer=AdamW(learning_rate=2e-5, weight_decay=1e-4), run_eagerly=False)
        
        self.train_dataset = self.__build_tf_dataset(train_dataset_path, batch_size)
        
        self.triplet_trainer.fit(
            self.train_dataset, 
            epochs=epochs
        )
        
        # Save the model
        self.triplet_trainer.embedding_model.transformer_model.save_pretrained(CONFIG['TRANSFORMER_SAVE_PATH'])
        print(f"Transformer model saved to {CONFIG['TRANSFORMER_SAVE_PATH']}")
    
    