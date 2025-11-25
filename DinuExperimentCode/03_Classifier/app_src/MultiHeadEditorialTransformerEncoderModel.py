import tensorflow as tf

import keras as K
from keras.layers import Dense
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC, SparseCategoricalAccuracy
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy 
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


class WarmupConstantSchedule(LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_ratio=0.1):
        """
        Initializes the schedule.

        Args:
            initial_lr (float): The target constant learning rate after warmup.
            total_steps (int): Total number of training steps.
            warmup_ratio (float): Fraction of total steps to use for warmup.
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

    def __call__(self, step):
        """
        Returns the learning rate for a given step.
        During the first `warmup_steps`, the learning rate increases linearly.
        After that, it remains constant.
        """
        step = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        # Use linear warmup if still in warmup phase.
        return tf.cond(
            step < warmup_steps_float,
            lambda: self.initial_lr * (step / warmup_steps_float),
            lambda: self.initial_lr
        )

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


class MultiHeadEditorialTransformerEncoderModel(K.Model):
    def __init__(self, transformer_model, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Load the transformer model
        self.transformer_model = transformer_model
        
        self.linear_layer1 = Dense(768, activation='relu')
        self.linear_layer2 = Dense(768, activation='relu')

        # head 1: multi-label (tags)
        self.statement_tags_classifier = Dense(num_classes, activation='sigmoid', name='statements_classifier')
        # head 2: multi-label (tags)
        self.editorial_tags_classifier = Dense(num_classes, activation='sigmoid', name='editorials_classifier')

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
        """
        inputs must be a dict containing:
           - statement_input_ids, statement_attention_mask
           - editorial_input_ids, editorial_attention_mask
        """
        def encode(input_ids, attention_mask):
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
            
            linear_output = self.linear_layer1(mean_pooled)
            linear_output = self.linear_layer2(linear_output)
            
            return linear_output
        
        stmt_repr = encode(
            inputs['statement_input_ids'],
            inputs['statement_attention_mask']
        )
        
        edit_repr = encode(
            inputs['editorial_input_ids'],
            inputs['editorial_attention_mask']
        )
        
        return {
            "tags_statement":       self.statement_tags_classifier(stmt_repr),
            "tags_editorial":       self.editorial_tags_classifier(edit_repr)
        }
    
    def compile_model(
        self, 
        run_eagerly=False, 
        threshold=0.5, 
        learning_rate=2e-5, 
        weight_decay=1e-4, 
        statement_loss_weight: float = 0.5,
        editorial_loss_weight: float = 0.5,
        total_steps=None
    ):
        
        if total_steps:
            # Create the learning rate schedule.
            lr_schedule = WarmupConstantSchedule(learning_rate, total_steps, warmup_ratio=0.1)

            # Define the optimizer, loss function and metrics
            optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
        else:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
                
        # both heads share the same loss type
        losses = {
            "tags_statement":   BinaryCrossentropy(name="loss_stmt"),
            "tags_editorial":   BinaryCrossentropy(name="loss_edit"),
        }
        loss_weights = {
            "tags_statement": statement_loss_weight,
            "tags_editorial": editorial_loss_weight,
        }
        
        if self.num_classes > 2:
            metrics = {
                "tags_statement": [
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
                ],
                "tags_editorial": [
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
                ],
            }
        else:
            metrics = {
                "tags_statement": [
                    # Macro Label Metrics
                    BinaryAccuracy(name='binary_accuracy', threshold=threshold), 
                    Precision(name='precision', thresholds=threshold), 
                    Recall(name='recall', thresholds=threshold), 
                    F1Score(name='f1', threshold=threshold),
                    # Area under the curve metrics
                    AUC(name='auc'),
                    AUC(name='prc_auc', curve='PR')
                ],
                "tags_editorial": [
                    # Macro Label Metrics
                    BinaryAccuracy(name='binary_accuracy', threshold=threshold), 
                    Precision(name='precision', thresholds=threshold), 
                    Recall(name='recall', thresholds=threshold), 
                    F1Score(name='f1', threshold=threshold),
                    # Area under the curve metrics
                    AUC(name='auc'),
                    AUC(name='prc_auc', curve='PR')
                ]
            }

        
        # Compile the model
        self.compile(
            optimizer=optimizer, 
            loss=losses, 
            loss_weights=loss_weights,
            metrics=metrics, 
            run_eagerly=run_eagerly
        )
