import tensorflow as tf

THRESHOLD = 0.5

class PrintScoresCallback(tf.keras.callbacks.Callback):
    def on_test_end(self, metric_names, logs):
        
        results = {}
        
        print("Evaluation Metrics:")
        for name, value in zip(metric_names, logs):
            if name == 'Label F1 Scores' or name == 'Label Accuracies':
                value = tf.keras.backend.get_value(value)
            print(f"{name}: {value}")
            results[name] = value
        
        return results

class PrintValidationScoresCallback(tf.keras.callbacks.Callback):
    def print_metrics(self, epoch, logs):
        # Define the metric names in the order they appear in the logs
        
        print(f'\nEpoch {epoch}: Validation Metrics:')
        for name, value in logs.items():
            if name.startswith('val_'):
                try:
                    value = tf.keras.backend.get_value(value)
                except Exception:
                    pass
                print(f"{name}: {value}")
    
    def on_epoch_end(self, epoch, logs=None):
        self.print_metrics(epoch + 1, logs)

    # def on_train_end(self, logs=None):
    #     self.print_metrics('Final', logs)

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        
@tf.keras.utils.register_keras_serializable()
class LabelWiseF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='label_wise_f1_score', num_labels=5, threshold=0.5, **kwargs):
        super(LabelWiseF1Score, self).__init__(name=name, **kwargs)
        self.num_labels = num_labels
        self.threshold = threshold
        
        # This weight will hold one F1 score per label.
        self.f1_scores = self.add_weight(name='f1_scores', shape=(num_labels,), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply the threshold to convert probabilities into binary predictions.
        y_pred_thresholded = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        f1_scores = []
        for i in range(self.num_labels):
            true_positives = tf.reduce_sum(y_true[:, i] * y_pred_thresholded[:, i])
            predicted_positives = tf.reduce_sum(y_pred_thresholded[:, i])
            actual_positives = tf.reduce_sum(y_true[:, i])
            
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
            
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        
        self.f1_scores.assign(tf.stack(f1_scores))

    def result(self):
        return self.f1_scores

    def reset_state(self):
        self.f1_scores.assign(tf.zeros_like(self.f1_scores))

@tf.keras.utils.register_keras_serializable()
class LabelWiseAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='label_wise_accuracy', num_labels=5, threshold=0.5, **kwargs):
        super(LabelWiseAccuracy, self).__init__(name=name, **kwargs)
        self.num_labels = num_labels
        self.threshold = threshold
        self.accuracies = self.add_weight(name='accuracies', shape=(num_labels,), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply the threshold
        y_pred_thresholded = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        accuracies = []
        for i in range(self.num_labels):
            correct_predictions = tf.equal(y_true[:, i], y_pred_thresholded[:, i])
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            accuracies.append(accuracy)
        
        self.accuracies.assign(tf.stack(accuracies))

    def result(self):
        return self.accuracies

    def reset_state(self):
        self.accuracies.assign(tf.zeros_like(self.accuracies))


@tf.keras.utils.register_keras_serializable()
def label_wise_f1_score(y_true, y_pred, threshold=THRESHOLD):
    # Apply threshold to get binary predictions.
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    f1_scores = []
    for i in range(y_true.shape[1]):
        true_positives = tf.reduce_sum(y_true[:, i] * y_pred_thresholded[:, i])
        predicted_positives = tf.reduce_sum(y_pred_thresholded[:, i])
        actual_positives = tf.reduce_sum(y_true[:, i])
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        f1_scores.append(f1)
    
    return f1_scores

@tf.keras.utils.register_keras_serializable()
def label_wise_macro_f1(y_true, y_pred, threshold=THRESHOLD):
    f1_scores = label_wise_f1_score(y_true, y_pred, threshold)
    macro_f1_score = tf.reduce_mean(tf.stack(f1_scores))
    return macro_f1_score

@tf.keras.utils.register_keras_serializable()
def subset_f1(y_true, y_pred, threshold=THRESHOLD):
    precision = subset_precision(y_true, y_pred, threshold)
    recall = subset_recall(y_true, y_pred, threshold)
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1

@tf.keras.utils.register_keras_serializable()
def label_wise_accuracy(y_true, y_pred, threshold=THRESHOLD):
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    accuracies = []
    for i in range(y_true.shape[1]):
        correct_predictions = tf.equal(y_true[:, i], y_pred_thresholded[:, i])
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        accuracies.append(accuracy)
    
    return accuracies

@tf.keras.utils.register_keras_serializable()
def label_wise_macro_accuracy(y_true, y_pred, threshold=THRESHOLD):
    accuracies = label_wise_accuracy(y_true, y_pred, threshold)
    macro_accuracy = tf.reduce_mean(tf.stack(accuracies))
    return macro_accuracy

@tf.keras.utils.register_keras_serializable()
def subset_accuracy(y_true, y_pred, threshold=THRESHOLD):
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    matches = tf.reduce_all(tf.equal(y_true, y_pred_thresholded), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

@tf.keras.utils.register_keras_serializable()
def subset_precision(y_true, y_pred, threshold=THRESHOLD):
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positive = tf.reduce_sum(tf.cast(tf.logical_and(y_pred_thresholded == 1, y_true == 1), tf.float32), axis=1)
    predicted_positive = tf.reduce_sum(tf.cast(y_pred_thresholded == 1, tf.float32), axis=1)
    precision = true_positive / (predicted_positive + tf.keras.backend.epsilon())
    return tf.reduce_mean(precision)

@tf.keras.utils.register_keras_serializable()
def subset_recall(y_true, y_pred, threshold=THRESHOLD):
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positive = tf.reduce_sum(tf.cast(tf.logical_and(y_pred_thresholded == 1, y_true == 1), tf.float32), axis=1)
    actual_positive = tf.reduce_sum(tf.cast(y_true == 1, tf.float32), axis=1)
    recall = true_positive / (actual_positive + tf.keras.backend.epsilon())
    return tf.reduce_mean(recall)
