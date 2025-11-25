
import os
import pandas as pd
import numpy as np
import ast

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer 

# local application/library specific imports
from app_config import AppConfig
from app_src import ClassifierChainWrapper
from app_src import CustomEncoder
from transformers import AutoTokenizer, TFAutoModel

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

class DecisionTreeEvaluator():
    
    def __init__(self) -> None:        
        self.estimator_collection = {}
        self.encoder_collection = []
        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        
        self.__define_models()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=512)

    def __define_models(self):
        logistic_regression_classifier = LogisticRegression(solver='newton-cg', random_state=RANDOM_STATE)
        kn_classifier = KNeighborsClassifier()
        decision_tree_classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        gaussian_nb_classifier = GaussianNB()
        random_forest_classifier = RandomForestClassifier(random_state=RANDOM_STATE)
        xgb_classifier = XGBClassifier(random_state=RANDOM_STATE)
        lgb_classifier = LGBMClassifier(random_state=RANDOM_STATE)
        svc_classifier = SVC(decision_function_shape='ovo')

        self.estimator_collection = {
            'LogisticRegression': logistic_regression_classifier,
            'KNeighborsClassifier': kn_classifier,
            'DecisionTreeClassifier': decision_tree_classifier,
            'GaussianNB': gaussian_nb_classifier,
            'RandomForestClassifier': random_forest_classifier,
            'XGBClassifier': xgb_classifier,
            'LGBMClassifier': lgb_classifier,
            'SVC': svc_classifier
        }
        
        self.encoder_collection = [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'sentence-transformers/multi-qa-distilbert-cos-v1',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'sentence-transformers/all-distilroberta-v1',
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'sentence-transformers/paraphrase-albert-small-v2',
            'microsoft/codebert-base',
            'roberta-base',
            'bert-base-uncased'
        ]


    def __read_test_data(self, number_of_tags, outside_dataset=False):
        if outside_dataset:
            self.test_dataset = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{number_of_tags}_TESTING_DATASET_PATH'])
        else:
            self.test_dataset = pd.read_csv(CONFIG[f'TOP_{number_of_tags}_TESTING_DATASET_PATH'])
        
    def __read_train_data(self, number_of_tags, outside_dataset=False):
        if outside_dataset:
            self.train_dataset = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{number_of_tags}_TRAINING_DATASET_PATH'])
        else:
            self.train_dataset = pd.read_csv(CONFIG[f'TOP_{number_of_tags}_TRAINING_DATASET_PATH'])
    
    def __encode_with_tfidf(self, problem_statements):
        return self.tfidf_vectorizer.fit_transform(problem_statements).toarray()

    def __encode_problem_statements(self, encoder_model, problem_statements, batch_size):
        if encoder_model == 'tfidf':
            return self.__encode_with_tfidf(problem_statements)
        else:
            return self.encoder.encode_problem_statement(problem_statements, batch_size)
    
    def __encode_tags(self, tags):        
        for idx, string_tag_list in enumerate(tags):
            tags[idx] = ast.literal_eval(string_tag_list)
        return np.array(tags)
    
    def __save_metrics(self, encoder, estimator_name, metrics_results, number_of_tags):
        
        scores = {key: str(value) for key, value in metrics_results.items()}
        csv_headers = ['Encoder Name', 'Estimator Name'] + list(scores.keys())
        output_data = [f'{encoder.name}', f'{estimator_name}'] + list(scores.values())

        # Create a DataFrame
        df = pd.DataFrame([output_data], columns=csv_headers)

        if not os.path.isfile(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH'], index=False, mode='a', header=False)
    
    def benchmark_estimators(self, encoder_batch_size, number_of_tags, estimator_name=None, transformer_name=None, transformer_model_path=None, outside_dataset=False):
        
        if transformer_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)            
            
            self.transformer_model = TFAutoModel.from_pretrained(transformer_model_path)
            
            self.encoder = CustomEncoder(transformer_name, self.transformer_model, self.tokenizer)
            print('Loaded transformer model from:', transformer_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
            self.transformer_model = TFAutoModel.from_pretrained(transformer_name)
            self.encoder = CustomEncoder(transformer_name, self.transformer_model, self.tokenizer)
            print('Loaded transformer model:', transformer_name)
        
        if outside_dataset:
            self.__read_train_data(number_of_tags, outside_dataset=True)
            self.__read_test_data(number_of_tags, outside_dataset=True)
        else:
            self.__read_train_data(number_of_tags)
            self.__read_test_data(number_of_tags)
        
        # Encode the problem statements and tags for training
        train_problem_statements = self.__encode_problem_statements(self.encoder, self.train_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
        train_tags = self.__encode_tags(self.train_dataset['problem_tags'].tolist())   
                              
        # Encode the problem statements and tags for testing
        test_problem_statements = self.__encode_problem_statements(self.encoder, self.test_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
        test_tags = self.__encode_tags(self.test_dataset['problem_tags'].tolist())
        
        # Start benchmarking the estimator models
        for name, estimator in self.estimator_collection.items():
            if estimator_name and name != estimator_name:
                continue
            
            print('Benchmarking estimator model:', name)
            
            classifierChainWrapper = ClassifierChainWrapper(estimator)
            
            classifierChainWrapper.fit(train_problem_statements, train_tags)

            metrics_results = classifierChainWrapper.predict(test_problem_statements, test_tags)
            
            self.__save_metrics(self.encoder, name, metrics_results, number_of_tags)
    
    def get_individual_predictions(self, encoder_batch_size, number_of_tags, estimator_name, transformer_name, transformer_model_path=None, outside_dataset=False, output_path=None):
        """
        Evaluates a model and returns individual probability predictions for each row and each class.
        
        Args:
            encoder_batch_size: Batch size for encoding
            number_of_tags: Number of tags in the dataset
            estimator_name: Name of the estimator to use
            transformer_name: Name of the transformer model
            transformer_model_path: Path to domain-adapted model (optional)
            outside_dataset: Whether to use outside dataset
            output_path: Path to save predictions CSV (optional)
        
        Returns:
            DataFrame with probability predictions for each row and each class
        """
        # Load encoder (same as benchmark_estimators)
        if transformer_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)            
            self.transformer_model = TFAutoModel.from_pretrained(transformer_model_path)
            self.encoder = CustomEncoder(transformer_name, self.transformer_model, self.tokenizer)
            print('Loaded transformer model from:', transformer_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
            self.transformer_model = TFAutoModel.from_pretrained(transformer_name)
            self.encoder = CustomEncoder(transformer_name, self.transformer_model, self.tokenizer)
            print('Loaded transformer model:', transformer_name)
        
        # Load test data
        if outside_dataset:
            self.__read_test_data(number_of_tags, outside_dataset=True)
        else:
            self.__read_test_data(number_of_tags)
        
        # Encode problem statements and tags
        test_problem_statements = self.__encode_problem_statements(self.encoder, self.test_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
        test_tags = self.__encode_tags(self.test_dataset['problem_tags'].tolist())
        
        # Get the estimator
        if estimator_name not in self.estimator_collection:
            raise ValueError(f"Estimator '{estimator_name}' not found. Available: {list(self.estimator_collection.keys())}")
        
        estimator = self.estimator_collection[estimator_name]
        print(f'Using estimator: {estimator_name}')
        
        # Train on training data (we need to load it for training)
        if outside_dataset:
            self.__read_train_data(number_of_tags, outside_dataset=True)
        else:
            self.__read_train_data(number_of_tags)
        
        train_problem_statements = self.__encode_problem_statements(self.encoder, self.train_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
        train_tags = self.__encode_tags(self.train_dataset['problem_tags'].tolist())
        
        # Train the model
        classifierChainWrapper = ClassifierChainWrapper(estimator)
        classifierChainWrapper.fit(train_problem_statements, train_tags)
        
        # Get probability predictions
        probability_predictions = classifierChainWrapper.predict_proba(test_problem_statements)
        
        # Create DataFrame with predictions
        results_df = self.test_dataset.copy()
        
        # Add probability columns for each class
        num_classes = probability_predictions.shape[1]
        for class_idx in range(num_classes):
            results_df[f'prob_class_{class_idx}'] = probability_predictions[:, class_idx]
        
        # Add ground truth columns for comparison
        for class_idx in range(num_classes):
            results_df[f'actual_class_{class_idx}'] = test_tags[:, class_idx]
        
        # Add a column showing which classes were actually present
        results_df['actual_classes'] = [
            [i for i, val in enumerate(row) if val == 1] 
            for row in test_tags
        ]
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f'Predictions saved to: {output_path}')
        
        return results_df