

import datetime
import os

class Config(object):
    """
    Config with all the paths and flags needed.
    """

    def __init__(self):
        
        WORKING_DIR = os.path.dirname(os.getcwd())
        BASE_DIR = os.path.dirname(os.path.abspath(WORKING_DIR))
        
        # Generate the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.config = {
            
            "WORKING_DIR": f"{WORKING_DIR}",
            "BASE_DIR": f"{BASE_DIR}",

            ############################################################################################################          
            
            "TOP_5_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_training_dataset.csv'),
            "TOP_10_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_training_dataset.csv'),
            "TOP_20_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_training_dataset.csv'),
            
            "TOP_5_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_testing_dataset.csv'),
            "TOP_10_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_testing_dataset.csv'),
            "TOP_20_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_testing_dataset.csv'),

            "TOP_5_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_validation_dataset.csv'), 
            "TOP_10_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_validation_dataset.csv'),
            "TOP_20_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_validation_dataset.csv'),

            ############################################################################################################          
            
            "TOP_5_TRAINING_ENHANCED_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_training_dataset.csv'),
            "TOP_10_TRAINING_ENHANCED_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_training_dataset.csv'),
            "TOP_20_TRAINING_ENHANCED_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_training_dataset.csv'),
            
            "TOP_5_TESTING_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_testing_dataset.csv'),
            "TOP_10_TESTING_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_testing_dataset.csv'),
            "TOP_20_TESTING_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_testing_dataset.csv'),

            "TOP_5_VALIDATION_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_5_validation_dataset.csv'), 
            "TOP_10_VALIDATION_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_10_validation_dataset.csv'),
            "TOP_20_VALIDATION_ENHANCED_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '04_DATASETS_ENHANCED_W_TAG_ENCODING', 'OUR_DATASET', 'top_20_validation_dataset.csv'),
            
            ################################################################################################################
            
            "OUTSIDE_TOP_5_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT5_train.csv'),
            "OUTSIDE_TOP_5_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT5_test.csv'),
            "OUTSIDE_TOP_5_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT5_validation.csv'),
        
            "OUTSIDE_TOP_10_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT10_train.csv'),
            "OUTSIDE_TOP_10_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT10_test.csv'),
            "OUTSIDE_TOP_10_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT10_validation.csv'),
            
            "OUTSIDE_TOP_20_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT20_train.csv'),
            "OUTSIDE_TOP_20_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT20_test.csv'),
            "OUTSIDE_TOP_20_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', '01_DATASETS_W_TAG_ENCODING', 'PSG_PREDICTING_ALGO', 'AMT20_validation.csv'),
            
            ############################################################################################################          

            "TOP_5_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_ChainClassifier_results.csv'),
            "TOP_10_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_ChainClassifier_results.csv'),
            "TOP_20_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_ChainClassifier_results.csv'),

            ############################################################################################################          

            "TOP_5_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_DecisionTree_onevsall_results.csv'),
            "TOP_10_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_DecisionTree_onevsall_results.csv'),
            "TOP_20_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_DecisionTree_onevsall_results.csv'), 
            
            ############################################################################################################
            "TOP_5_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_SingleModelClassifier_results.csv'),
            "TOP_10_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_SingleModelClassifier_results.csv'),
            "TOP_20_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_SingleModelClassifier_results.csv'),
            
            ############################################################################################################
            "TOP_5_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_OneVsAll_results.csv'),
            "TOP_10_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_OneVsAll_results.csv'),
            "TOP_20_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_OneVsAll_results.csv'),
            
            ############################################################################################################
            "TOP_5_MULTIHEADDIFF_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_MultiHeadDiffSingleModelClassifier_results.csv'),
            "TOP_10_MULTIHEADDIFF_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_MultiHeadDiffSingleModelClassifier_results.csv'),
            "TOP_20_MULTIHEADDIFF_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_MultiHeadDiffSingleModelClassifier_results.csv'),
            
            ############################################################################################################
            "TOP_5_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_5_MultiHeadEditorialSingleModelClassifier_results.csv'),
            "TOP_10_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_10_MultiHeadEditorialSingleModelClassifier_results.csv'),
            "TOP_20_MULTIHEADEDITORIAL_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', 'top_20_MultiHeadEditorialSingleModelClassifier_results.csv'),
            
            ############################################################################################################          
            "TOP_5_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS', 'OUR_DATASET', 'top_5_nli_training_dataset.csv'),
            "TOP_10_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS', 'OUR_DATASET', 'top_10_nli_training_dataset.csv'),
            "TOP_20_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS', 'OUR_DATASET', 'top_20_nli_training_dataset.csv'),
            
            "TOP_5_BASIC_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'OUR_DATASET', 'top_5_nli_training_dataset.csv'),
            "TOP_10_BASIC_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'OUR_DATASET', 'top_10_nli_training_dataset.csv'),
            "TOP_20_BASIC_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'OUR_DATASET', 'top_20_nli_training_dataset.csv'),
                        
            ############################################################################################################
            
            "OUTSIDE_TOP_5_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS',  'PSG_PREDICTING_ALGO', 'AMT5_nli_train.csv'),
            "OUTSIDE_TOP_10_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS',  'PSG_PREDICTING_ALGO', 'AMT10_nli_train.csv'),
            "OUTSIDE_TOP_20_NLI_TRAINING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'AUGMENTED_NLI_DATASETS',  'PSG_PREDICTING_ALGO', 'AMT20_nli_train.csv'),
            
            "OUTSIDE_TOP_5_BASIC_NLI_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'PSG_PREDICTING_ALGO', 'AMT5_nli_train.csv'),
            "OUTSIDE_TOP_10_BASIC_NLI_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'PSG_PREDICTING_ALGO', 'AMT10_nli_train.csv'),
            "OUTSIDE_TOP_20_BASIC_NLI_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '04_NLI_Datasets', 'BASIC_NLI_DATASETS', 'PSG_PREDICTING_ALGO', 'AMT20_nli_train.csv'),
            
            ############################################################################################################
            "TRANSFORMER_SAVE_PATH": os.path.join(BASE_DIR, '05_MODELS', '01_DomainAdaped_Models', f'transformer_model_{current_time}'),
            "MODEL_SAVE_PATH_ROOT": os.path.join(BASE_DIR, '05_MODELS', '01_DomainAdaped_Models'),
            "TRANSFORMER_SAVE_PATH_ROOT": os.path.join(BASE_DIR, '05_MODELS', '01_DomainAdaped_Models')
        }   

        self.global_constants = {
            "RANDOM_SEED": 42
        }
        
    def return_global_constants(self):
        """
        Return global constants
        Returns:
            None
        """
        return self.global_constants
    
    def return_config(self):
        """
        Return entire config dictionary
        Returns:
            None
        """
        return self.config
