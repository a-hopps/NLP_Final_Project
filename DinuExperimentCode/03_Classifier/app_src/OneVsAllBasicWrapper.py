# standard library imports
import os
import numpy as np
import copy

# related third-party
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score

# local application/library specific imports
from app_config import AppConfig
from app_src.CustomMetrics import subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_accuracy, label_wise_f1_score

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']


class OneVsAllBasicWrapper():
    def __init__(self, base_estimator, number_of_labels):
        self.models = []
        self.number_of_labels = number_of_labels
        self.base_estimator = base_estimator
        
            
    def fit(self, train_problem_statements, train_tags, validation_problem_statements=None, validation_tags=None):
        
        for label_idx in range(self.number_of_labels):
            train_tags_i = train_tags[:, label_idx]
            estimator = copy.deepcopy(self.base_estimator)
            estimator.fit(train_problem_statements, train_tags_i)
            self.models.append(estimator)
    
    def predict(self, test_problem_statements_embeddings, test_tags):
        # We'll create an array to hold predictions for all 5 labels
        predictions = np.zeros((test_problem_statements_embeddings.shape[0], self.number_of_labels), dtype=int)
        
        for label_idx, estimator in enumerate(self.models):
            pred_i = estimator.predict(test_problem_statements_embeddings)  # shape: (n_test_samples,)
            predictions[:, label_idx] = pred_i
        
        # # Make predictions on the test data
        # predictions = self.classifier_chain.predict(test_problem_statements_embeddings)
        # # Transform from sparce array to dense array
        # predictions_dense = predictions.toarray()
        # # Transform from float to integer
        # predictions = np.round(predictions_dense).astype(int)
        
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
