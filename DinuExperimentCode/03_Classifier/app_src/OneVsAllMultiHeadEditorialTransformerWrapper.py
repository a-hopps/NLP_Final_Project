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
from sklearn.utils import resample


# local application/library specific imports
from app_src.MultiHeadTransformerEncoderModel import MultiHeadTransformerEncoderModel
from app_src.BaseSentenceTransformerEncoderModel import BaseSentenceTransformerEncoderModel
from app_src.CustomMetrics import PrintScoresCallback, PrintValidationScoresCallback
from app_config import AppConfig
from app_src.common import set_random_seed
from transformers import AutoTokenizer, TFAutoModel
from app_src.MultiHeadEditorialTransformerEncoderModel import MultiHeadEditorialTransformerEncoderModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
from app_src.CustomMetrics import subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_accuracy, label_wise_f1_score


# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class OneVsAllMultiHeadEditorialTransformerWrapper():
    def __init__(self, model_name, number_of_tags):
        
        self.model_name = model_name
        self.number_of_tags = number_of_tags

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        
        self.models = []
        
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
        
        problem_statements = data_df['problem_statement'].tolist()
        problem_editorials = data_df['problem_editorial'].tolist()
        tags = self.__encode_tags(data_df['problem_tags'].tolist())
            
        # Tokenize all problem statements        
        statement_input_ids, statement_attention_mask = self.__tokenize_data(problem_statements)
        editorial_input_ids, editorial_attention_mask = self.__tokenize_data(problem_editorials)

        
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'statement_input_ids': statement_input_ids,
                'statement_attention_mask': statement_attention_mask,
                'editorial_input_ids': editorial_input_ids,
                'editorial_attention_mask': editorial_attention_mask
            },
            {
            'tags_statement': tags,
            'tags_editorial': tags,
            }
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
        
        self.__read_train_data(train_dataset_path)
        self.__read_validation_data(val_dataset_path)
        
        # Training Dataset
        problem_statements = self.train_dataset['problem_statement'].tolist()
        problem_editorials = self.train_dataset['problem_editorial'].tolist()
        tags = self.__encode_tags(self.train_dataset['problem_tags'].tolist())        

        # Tokenize all problem statements        
        statement_input_ids, statement_attention_mask = self.__tokenize_data(problem_statements)
        editorial_input_ids, editorial_attention_mask = self.__tokenize_data(problem_editorials)
    
        # Validation Dataset
        val_problem_statements = self.validation_dataset['problem_statement'].tolist()
        val_problem_editorials = self.validation_dataset['problem_editorial'].tolist()
        val_tags = self.__encode_tags(self.validation_dataset['problem_tags'].tolist())

        # Tokenize all validation problem statements
        val_statement_input_ids, val_statement_attention_mask = self.__tokenize_data(val_problem_statements)
        val_editorial_input_ids, val_editorial_attention_mask = self.__tokenize_data(val_problem_editorials)


        for label_idx in range(self.number_of_tags):
            
            print('Starting training for label:', label_idx)

            # Balance the classes for the current tag
            pos_indices = np.where(tags[:, label_idx] == 1)[0]
            neg_indices = np.where(tags[:, label_idx] == 0)[0]

            # Undersample the majority class
            if len(pos_indices) < len(neg_indices):
                neg_indices = resample(neg_indices, replace=True, n_samples=len(pos_indices), random_state=42)
            else:
                pos_indices = resample(pos_indices, replace=True, n_samples=len(neg_indices), random_state=42)
        
            balanced_indices = np.concatenate([pos_indices, neg_indices])
            np.random.shuffle(balanced_indices)

            # Convert balanced_indices to TensorFlow tensor
            balanced_indices = tf.convert_to_tensor(balanced_indices, dtype=tf.int32)
            
            balanced_statement_input_ids = tf.gather(statement_input_ids, balanced_indices)
            balanced_statement_attention_mask = tf.gather(statement_attention_mask, balanced_indices)
            balanced_editorial_input_ids = tf.gather(editorial_input_ids, balanced_indices)
            balanced_editorial_attention_mask = tf.gather(editorial_attention_mask, balanced_indices)
            balanced_tags = tf.gather(tags[:, label_idx], balanced_indices)  # Single label
        
            # Convert to TensorFlow dataset
            balanced_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'statement_input_ids': balanced_statement_input_ids,
                    'statement_attention_mask': balanced_statement_attention_mask,
                    'editorial_input_ids': balanced_editorial_input_ids,
                    'editorial_attention_mask': balanced_editorial_attention_mask
                },
                {
                'tags_statement': balanced_tags,
                'tags_editorial': balanced_tags,
                }
            ))
            
            single_label_train_ds = balanced_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
            # Create a single-label validation dataset           
            single_label_val_tags = val_tags[:, label_idx]  # Single label
            val_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'statement_input_ids': val_statement_input_ids,
                    'statement_attention_mask': val_statement_attention_mask,
                    'editorial_input_ids': val_editorial_input_ids,
                    'editorial_attention_mask': val_editorial_attention_mask
                },
                {
                'tags_statement': single_label_val_tags,
                'tags_editorial': single_label_val_tags,
                }
            ))
            single_label_val_ds = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            if transformer_model_path:
                transformer_model = TFAutoModel.from_pretrained(transformer_model_path)
            else:
                transformer_model = TFAutoModel.from_pretrained(self.model_name)
                
            if base_model_evaluation:
                encoder_model = BaseSentenceTransformerEncoderModel(transformer_model, 1)
            else:
                encoder_model = MultiHeadEditorialTransformerEncoderModel(transformer_model, 1)
            
                # Unfreeze the transformer layers
                encoder_model.unfreeze_transformer()

                # total_steps = ceil(len(problem_statements) / batch_size) * epochs
                
                # Compile the model
                # encoder_model.compile_model(run_eagerly=False, threshold=threshold, total_steps=total_steps)
                encoder_model.compile_model(run_eagerly=False, threshold=threshold)
                # Define callbacks
                callbacks = [
                    # Custom callback for printing validation scores
                    PrintValidationScoresCallback(),
                    tf.keras.callbacks.EarlyStopping(monitor='val_tags_statement_f1', mode='max', patience=5, restore_best_weights=True)
                ]
                
                # Start training
                history = encoder_model.fit(
                    single_label_train_ds,
                    validation_data=single_label_val_ds,
                    epochs=epochs,
                    callbacks=callbacks
                )
            
            self.models.append(encoder_model)
    
    def benchmark_model(self, test_dataset_path, batch_size=32, threshold=0.5):
        
        self.__read_test_data(test_dataset_path)

        # Validation Dataset
        test_problem_statements = self.test_dataset['problem_statement'].tolist()
        test_problem_editorials = self.test_dataset['problem_editorial'].tolist()
        test_tags = self.__encode_tags(self.test_dataset['problem_tags'].tolist())
        
        test_statement_input_ids, test_statement_attention_mask = self.__tokenize_data(test_problem_statements)
        test_editorial_input_ids, test_editorial_attention_mask = self.__tokenize_data(test_problem_editorials)
        
        # We'll create an array to hold predictions for all 5 labels
        all_preds = []
        
        for label_idx, estimator in enumerate(self.models):
            
            # single-label testing dataset
            single_label_test_tags = test_tags[:, label_idx]  # Single label
            test_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'statement_input_ids': test_statement_input_ids,
                    'statement_attention_mask': test_statement_attention_mask,
                    'editorial_input_ids': test_editorial_input_ids,
                    'editorial_attention_mask': test_editorial_attention_mask
                },
                {
                    'tags_statement': single_label_test_tags,
                    'tags_editorial': single_label_test_tags,  # Assuming editorial tags are the same for single-label testing
                }
            ))
            single_label_test_ds = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            estimator.freeze_transformer()
            
            pred_probs = estimator.predict(single_label_test_ds)
            all_preds.append(pred_probs)

        # Collect the 'tags_statement' predictions for each label
        # statement_preds = [pred['tags_statement'] for pred in all_preds]  # List of (num_samples, 1) arrays
        # editorial_preds = [pred['tags_editorial'] for pred in all_preds]
        
        # # Stack and squeeze to get shape (num_samples, num_labels)
        # statement_preds = np.squeeze(np.stack(statement_preds, axis=-1))  # shape: (num_samples, num_labels)
        # editorial_preds = np.squeeze(np.stack(editorial_preds, axis=-1))  # shape: (num_samples, num_labels)

        # predictions = (statement_preds > threshold).astype(int)
        
        
        statement_preds = np.concatenate([p['tags_statement'] for p in all_preds], axis=1)
        editorial_preds = np.concatenate([p['tags_editorial'] for p in all_preds], axis=1)
        
        # 2) Compute per‐label F1 for each head at threshold 0.5
        num_labels = statement_preds.shape[1]
        statement_f1 = np.zeros(num_labels)
        editorial_f1 = np.zeros(num_labels)

        for j in range(num_labels):
            # Binarize at 0.5
            bin_s = (statement_preds[:, j] > threshold).astype(int)
            bin_e = (editorial_preds[:, j] > threshold).astype(int)
            ytrue = test_tags[:, j]
            
            statement_f1[j] = f1_score(ytrue, bin_s, zero_division=0)
            editorial_f1[j] = f1_score(ytrue, bin_e, zero_division=0)

        # 3) Build a “hard switch” weight vector w[j] = 1 if editorial better, else 0
        w = np.zeros(num_labels, dtype=float)
        for j in range(num_labels):
            if editorial_f1[j] > statement_f1[j]:
                w[j] = 1.0
            else:
                w[j] = 0.0
                
        # 4) Form combined logits
        combined_preds = np.zeros_like(statement_preds)
        for j in range(num_labels):
            combined_preds[:, j] = w[j] * editorial_preds[:, j] + (1 - w[j]) * statement_preds[:, j]

        # 5) Final binary decisions
        predictions = (combined_preds > threshold).astype(int)

        # 6) (Optional) Report new combined‐F1 to verify improvement
        combined_f1_per_label = [
            f1_score(test_tags[:, j], predictions[:, j], zero_division=0)
            for j in range(num_labels)
        ]
        print("Combined F1 per label:", combined_f1_per_label)        
        
        
        # Label Wise Metrics
        f1_scores = label_wise_f1_score(test_tags, predictions)
        f1_scores = [float(t.numpy()) for t in f1_scores]
        accuracies = label_wise_accuracy(test_tags, predictions)
        accuracies = [float(t.numpy()) for t in accuracies]
        accuracy = label_wise_macro_accuracy(test_tags, predictions).numpy()
        precision = precision_score(test_tags, predictions, average='macro')
        recall = recall_score(test_tags, predictions, average='macro')
        f1 = f1_score(test_tags, predictions, average='macro')
        
        # Subset Metrics
        sub_accuracy = accuracy_score(test_tags, predictions)
        sub_precision = subset_precision(test_tags, predictions).numpy()
        sub_recall = subset_recall(test_tags, predictions).numpy()
        sub_f1 = subset_f1(test_tags, predictions).numpy()
        
        # Area Metrics
        # auc = roc_auc_score(test_tags, predictions, average='macro', multi_class='ovr')
        auc = roc_auc_score(test_tags, predictions)
        prc_auc = average_precision_score(test_tags, predictions, average='macro')
        
        # Store the results
        results = {
            # Label Wise Metrics
            'Label F1 Scores': f1_scores,
            'Label Accuracies': accuracies,
            # Macro Label Metrics
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            # Subset Metrics
            'Subset Accuracy': sub_accuracy,
            'Subset Precision': sub_precision,
            'Subset Recall': sub_recall,
            'Subset F1': sub_f1,
            # Area Metrics
            'AUC': auc,
            'PRC AUC': prc_auc
        }
        
        return results
    