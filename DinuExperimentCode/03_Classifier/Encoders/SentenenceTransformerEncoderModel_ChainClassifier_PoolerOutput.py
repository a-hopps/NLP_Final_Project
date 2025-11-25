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

class SentenceTransformerEncoderChainModel(K.Model):
    def __init__(self, model_name, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes

        # Load the transformer model
        self.transformer = TFAutoModel.from_pretrained(model_name)
        
        # Create a classifier head for each label.
        # Each head outputs a single probability.
        self.classifier_heads = [Dense(1, activation='sigmoid') for _ in range(num_classes)]


    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'num_classes': self.num_classes
        })
        return config

    def freeze_transformer(self):
        """Freeze the transformer layers to prevent them from training."""
        self.transformer.trainable = False

    def unfreeze_transformer(self):
        """Unfreeze the transformer layers to allow fine-tuning."""
        self.transformer.trainable = True

    def call(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Print shapes for debugging
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Attention mask shape: {attention_mask.shape}")
        
        # Get the transformer outputs. The first element is usually the last hidden states.
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token) for classification.
        cls_output = transformer_outputs[0][:, 0, :]  # shape: (batch_size, hidden_size)
        
        # print(f"Transformer embeddings shape: {transformer_output.shape}")
        
        # Pass the pooled output through the classifier
        predictions = []
        chain_input = cls_output

        # Iterate over the classifier heads in a chain.
        for head in self.classifier_heads:
            # Predict the current label.
            pred = head(chain_input)
            predictions.append(pred)
            # Append the current prediction to the shared features for the next classifier.
            # This creates the "chain" effect.
            chain_input = tf.concat([chain_input, pred], axis=-1)
        
        # Concatenate all predictions to form a (batch_size, num_labels) output.
        return tf.concat(predictions, axis=-1)
    
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
