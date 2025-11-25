import os
import pandas as pd
import numpy as np
import ast
import random
import tensorflow as tf
from transformers import AutoTokenizer
from math import ceil
import keras as K
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# local application/library specific imports
from app_src.MultiHeadTransformerEncoderModel import MultiHeadTransformerEncoderModel
from app_src.BaseSentenceTransformerEncoderModel import BaseSentenceTransformerEncoderModel
from app_src.CustomMetrics import PrintScoresCallback, PrintValidationScoresCallback
from app_config import AppConfig
from app_src.common import set_random_seed
from transformers import AutoTokenizer, TFAutoModel

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class MultiHeadTransformerWrapper():
    def __init__(self, model_name, number_of_tags, number_of_difficulty_tags):
        
        self.model_name = model_name
        self.number_of_tags = number_of_tags
        self.number_of_difficulty_tags = number_of_difficulty_tags

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        
        set_random_seed(RANDOM_STATE)
        
    def __read_test_data(self, test_dataset_path):
        self.test_dataset = pd.read_csv(test_dataset_path)
        
    def __read_train_data(self, train_dataset_path):
        self.train_dataset = pd.read_csv(train_dataset_path)
    
    def __read_validation_data(self, val_dataset_path):
        self.validation_dataset = pd.read_csv(val_dataset_path)
    
    def __encode_tags(self, tags):
        for idx, string_tag_list in enumerate(tags):
            tags[idx] = ast.literal_eval(string_tag_list)
        tags = np.array(tags)
        # Ensure tags are in a consistent format (e.g., a NumPy array)
        if not isinstance(tags, (np.ndarray, tf.Tensor)):
            tags = np.array(tags)
        
        return tags
    
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
    
    def __build_tf_dataset(self, data_df, batch_size = 32, shuffle_buffer_size=10000):
        # Use the instance encoder if it's available; otherwise, create a new one.
        if hasattr(self, 'label_encoder'):
            label_encoder = self.label_encoder
        else:
            label_encoder = LabelEncoder()
        
        problem_statements = data_df['problem_statement'].tolist()
        tags = self.__encode_tags(data_df['problem_tags'].tolist())
        difficulties = label_encoder.transform(data_df['problem_dificulty'])
        difficulties = difficulties.astype(np.int32)
        # print(difficulties)
            
        # Tokenize all problem statements
        input_ids, attention_mask = self.__tokenize_data(problem_statements)

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(tags.shape)        
        # print(len(difficulties))
        
        labels = {
            "tags": tags,
            "difficulty": difficulties
        }
        
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            labels
        ))
                
        # Apply optimizations: shuffle, cache, batch, prefetch
        tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)
        tf_dataset = tf_dataset.cache()  # Use caching if your dataset fits in memory; otherwise, consider file-based caching.
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Print the shapes of the batches to verify
        # for batch in dataset.take(1):
        #     input_batch, tag_batch = batch
        #     print(f"Input batch shape: {input_batch['input_ids'].shape}")
        #     print(f"Attention mask batch shape: {input_batch['attention_mask'].shape}")
        #     print(f"Tags batch shape: {tag_batch.shape}")
        
        return tf_dataset
    
    def train_model(self, train_dataset_path, val_dataset_path, epochs=5, batch_size=32, threshold=0.5, transformer_model_path=None, base_model_evaluation=False):
        
        if transformer_model_path:
            self.transformer_model = TFAutoModel.from_pretrained(transformer_model_path)
        else:
            self.transformer_model = TFAutoModel.from_pretrained(self.model_name)
        
        if base_model_evaluation:
            self.encoder_model = BaseSentenceTransformerEncoderModel(self.transformer_model, self.number_of_tags)
            
            # Unfreeze the transformer layers
            self.encoder_model.unfreeze_transformer()
    
            # Compile the model
            self.encoder_model.compile_model(run_eagerly=False, threshold=threshold)
            
        else:
            self.encoder_model = MultiHeadTransformerEncoderModel(self.transformer_model, self.number_of_tags, self.number_of_difficulty_tags)
            
            self.__read_train_data(train_dataset_path)
            self.__read_validation_data(val_dataset_path)

            # Fit label encoder on training data and save it as an instance variable
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit_transform(self.train_dataset['problem_dificulty'])
            
            self.train_dataset  = self.__build_tf_dataset(self.train_dataset, batch_size)
            self.validation_dataset = self.__build_tf_dataset(self.validation_dataset, batch_size)
            
            # Unfreeze the transformer layers
            self.encoder_model.unfreeze_transformer()
    
            # Compile the model
            self.encoder_model.compile_model(run_eagerly=False, threshold=threshold)
        
            # Define callbacks
            callbacks = [
                # Custom callback for printing validation scores
                PrintValidationScoresCallback(),
                # Early stopping callback
                tf.keras.callbacks.EarlyStopping(monitor='val_tags_label_wise_macro_f1', mode='max', patience=5, restore_best_weights=True)
            ]
            
            # Start training
            history = self.encoder_model.fit(
                self.train_dataset,
                validation_data=self.validation_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
        
        # Save the model
        # self.encoder_model.save_weights(CONFIG['MODEL_SAVE_PATH'])
        # print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")
    
    def benchmark_model(self, test_dataset_path, batch_size=32, threshold=0.5):
        
        self.__read_test_data(test_dataset_path)

        # Convert test data to tf.data.Dataset
        self.test_dataset = self.__build_tf_dataset(self.test_dataset, batch_size)
            
        self.encoder_model.freeze_transformer()

        # Evaluate the model
        logs = self.encoder_model.evaluate(self.test_dataset, return_dict=True)
        
        metrics_values = [
            logs['loss'],
            logs['tags_loss'],
            logs['difficulty_loss'],
            # Label Wise Metrics
            logs['tags_label_wise_f1_score'],
            logs['tags_label_wise_accuracy'],
            # Macro Label Metrics
            logs['tags_binary_accuracy'],
            logs['tags_precision'],
            logs['tags_recall'],
            logs['tags_label_wise_macro_f1'],
            # Subset Metrics
            logs['tags_subset_accuracy'],
            logs['tags_subset_precision'],
            logs['tags_subset_recall'],
            logs['tags_subset_f1'],
            # Area Metrics
            logs['tags_auc'],
            logs['tags_prc_auc'],
            # Difficulty Metrics
            logs['difficulty_acc_diff']
        ]

        # Assuming these are the metric names in the same order as the results                
        metric_names = [
            'Overall Loss',
            'Tags Loss',
            'Difficulty Loss',
            # Label Wise Metrics
            'Label F1 Scores',
            'Label Accuracies',
            # Macro Label Metrics
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            # Subset Metrics
            'Subset Accuracy',
            'Subset Precision',
            'Subset Recall',
            'Subset F1',
            # Area Metrics
            'AUC',
            'PRC AUC',
            # Difficulty Metrics
            'Difficulty Accuracy'
        ]
        
        # Manually invoke the PrintScoresCallback to print the validation metrics
        callback = PrintScoresCallback()
        evaluation_results = callback.on_test_end(metric_names=metric_names, logs=metrics_values)
        return evaluation_results
    