
import keras as K
from transformers import AutoTokenizer
import tensorflow as tf

class NLITransformerEmbeddingModel(K.Model):
    def __init__(self, transformer, **kwargs):
        super().__init__(**kwargs)
        self.transformer_model = transformer

    def freeze_transformer(self):
        """Freeze the transformer layers to prevent them from training."""
        self.trainable = False
        self.transformer_model.trainable = False

    def unfreeze_transformer(self):
        """Unfreeze the transformer layers to allow fine-tuning."""
        self.trainable = True
        self.transformer_model.trainable = True

    def call(self, inputs, training=True):
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        transformer_outputs = self.transformer_model(
            input_ids,
            attention_mask,
            training=training
        )
        
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

        return mean_pooled

