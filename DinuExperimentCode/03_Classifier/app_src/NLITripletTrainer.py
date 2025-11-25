import keras as K
import tensorflow as tf

def triplet_margin_loss(anchor_emb, pos_emb, neg_emb, margin=1.0):
    # Normalize the embeddings
    anchor_emb = tf.nn.l2_normalize(anchor_emb, axis=1)
    pos_emb = tf.nn.l2_normalize(pos_emb, axis=1)
    neg_emb = tf.nn.l2_normalize(neg_emb, axis=1)
    
    # Calculate cosine similarity
    pos_sim = tf.reduce_sum(anchor_emb * pos_emb, axis=1)  # shape (batch,)
    neg_sim = tf.reduce_sum(anchor_emb * neg_emb, axis=1)
    
    # Convert cosine similarity to cosine distance
    pos_dist = 1 - pos_sim
    neg_dist = 1 - neg_sim
    
    # Calculate the triplet margin loss
    basic_loss = margin + pos_dist - neg_dist
    loss = tf.reduce_mean(tf.nn.relu(basic_loss))
    return loss

class NLITripletTrainer(K.Model):
    def __init__(self, embedding_model, margin=1.0):
        super().__init__()
        self.embedding_model = embedding_model
        self.margin = margin
        
    def call(self, inputs, training=True):
        # Define the forward pass
        anchor, positive, negative = inputs
        anchor_emb = self.embedding_model(anchor, training=training)
        positive_emb = self.embedding_model(positive, training=training)
        negative_emb = self.embedding_model(negative, training=training)
        return anchor_emb, positive_emb, negative_emb
    
    def train_step(self, data):
        anchor, positive, negative = data

        with tf.GradientTape() as tape:
            anchor_emb, positive_emb, negative_emb = self((anchor, positive, negative), training=True)
            loss_value = triplet_margin_loss(anchor_emb, positive_emb, negative_emb)

        # # Print the embeddings for debugging
        # print("Anchor Embeddings:", anchor_emb.numpy())
        # print("Positive Embeddings:", positive_emb.numpy())
        # print("Negative Embeddings:", negative_emb.numpy())
        
        # Print shapes for better readability
        print(f"Anchor Embeddings Shape: {anchor_emb.shape}")
        print(f"Positive Embeddings Shape: {positive_emb.shape}")
        print(f"Negative Embeddings Shape: {negative_emb.shape}")
        print(f"Loss Value: {loss_value}")
        
        print(self.embedding_model.transformer_model.trainable_variables)
        # print(self.embedding_model.transformer_model.trainable_weights)
        
        # Compute gradients
        grads = tape.gradient(loss_value, self.embedding_model.transformer_model.trainable_variables)

        print(f"Gradients: {grads}")
        print(f"Trainable Variables: {self.embedding_model.transformer_model.trainable_variables}")        
        
        self.optimizer.apply_gradients(zip(grads, self.embedding_model.transformer_model.trainable_variables))
        return {"loss": loss_value}
    