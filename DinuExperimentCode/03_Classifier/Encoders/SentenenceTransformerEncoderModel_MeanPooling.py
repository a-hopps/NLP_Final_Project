import tensorflow as tf

import keras as K
from keras.layers import Dense
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from keras.optimizers import AdamW
from transformers import AutoTokenizer, TFAutoModel

import random

# local application/library specific imports
from app_config import AppConfig
from app_src.CustomMetrics import subset_accuracy, subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_macro_f1, LabelWiseF1Score, LabelWiseAccuracy

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class SentenceTransformerEncoderModel(K.Model):
    def __init__(self, transformer_model, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        # Load the transformer model
        self.transformer_model = transformer_model
        
        # The classifier layer uses a sigmoid activation for multi-label classification.
        self.classifier = Dense(num_classes, activation='sigmoid')

    def freeze_transformer(self):
        """Freeze the transformer layers to prevent them from training."""
        self.transformer_model.trainable = False

    def unfreeze_transformer(self):
        """Unfreeze the transformer layers to allow fine-tuning."""
        self.transformer_model.trainable = True
        
    def get_embeddings(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        transformer_outputs = self.transformer_model(input_ids, attention_mask=attention_mask)
        return transformer_outputs.last_hidden_state
    
    def call(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Print shapes for debugging
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Attention mask shape: {attention_mask.shape}")
        
        # Get the transformer outputs. The first element is usually the last hidden states.
        transformer_outputs = self.transformer_model(input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden states (batch_size, seq_len, hidden_dim)
        hidden_states = transformer_outputs[0]

        # Convert attention_mask to float so we can do elementwise multiplication
        # shape: (batch_size, seq_len)
        mask = tf.cast(attention_mask, dtype=hidden_states.dtype)

        # Expand the mask for broadcast: (batch_size, seq_len, 1)
        mask = tf.expand_dims(mask, axis=-1)

        # Sum up the token embeddings * mask along the seq_len dimension
        # shape: (batch_size, hidden_dim)
        masked_sum = tf.reduce_sum(hidden_states * mask, axis=1)

        # Avoid division by zero by forcing at least one non-zero denominator
        mask_sum = tf.reduce_sum(mask, axis=1)  # shape: (batch_size, 1)
        mask_sum = tf.clip_by_value(mask_sum, clip_value_min=1e-9, clip_value_max=1e9)

        # Mean pooling: divide the summed embeddings by the number of valid tokens
        mean_pooled = masked_sum / mask_sum  # shape: (batch_size, hidden_dim)

        # Pass the mean-pooled embedding through your classifier
        logits = self.classifier(mean_pooled)
                
        return logits
    
    def compile_model(self, run_eagerly=False, threshold=0.5, learning_rate=2e-5, weight_decay=1e-4):
        # Define the optimizer, loss function and metrics
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        loss = BinaryCrossentropy()
        metrics = [
            # Label label-wise Metrics
            LabelWiseF1Score(name='label_wise_f1_score', threshold=threshold),
            LabelWiseAccuracy(name='label_wise_accuracy', threshold=threshold),
            # Macro Label Metrics
            BinaryAccuracy(name='binary_accuracy', threshold=threshold), 
            Precision(name='precision', thresholds=threshold), 
            Recall(name='recall', thresholds=threshold), 
            label_wise_macro_f1,
            # Subset Metrics
            subset_accuracy,
            subset_precision,
            subset_recall,
            subset_f1,
            # Area under the curve metrics
            AUC(name='auc'),
            AUC(name='prc_auc', curve='PR')
            ]
        
        # Compile the model
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
