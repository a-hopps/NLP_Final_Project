import tensorflow as tf

import keras as K
from keras.layers import Dense, Dropout
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from keras.optimizers import AdamW
from transformers import AutoTokenizer, TFAutoModel
from keras.optimizers.schedules import LearningRateSchedule

import random

# local application/library specific imports
from app_config import AppConfig
from app_src.CustomMetrics import subset_accuracy, subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_macro_f1, F1Score, LabelWiseF1Score, LabelWiseAccuracy

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)


class BaseSentenceTransformerEncoderModel(K.Model):
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
        
        # Get the transformer outputs. The first element is usually the last hidden states.
        transformer_outputs = self.transformer_model(input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token) for classification.
        cls_output = transformer_outputs[0][:, 0, :]  # shape: (batch_size, hidden_size)
        
        # Pass the pooled output through the classifier
        logits = self.classifier(cls_output)
                
        return logits
    
    def compile_model(self, run_eagerly=False, threshold=0.5, learning_rate=2e-5, weight_decay=1e-4, total_steps=None):
        
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
        loss = BinaryCrossentropy()

        if self.num_classes > 2:
            metrics = [
                # Label label-wise Metrics
                LabelWiseF1Score(name='label_wise_f1_score', threshold=threshold, num_labels=self.num_classes),
                LabelWiseAccuracy(name='label_wise_accuracy', threshold=threshold, num_labels=self.num_classes),
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
        else:
            metrics = [
                # Macro Label Metrics
                BinaryAccuracy(name='binary_accuracy', threshold=threshold), 
                Precision(name='precision', thresholds=threshold), 
                Recall(name='recall', thresholds=threshold), 
                F1Score(name='f1', threshold=threshold),
                # Area under the curve metrics
                AUC(name='auc'),
                AUC(name='prc_auc', curve='PR')
                ]
        
        # Compile the model
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
