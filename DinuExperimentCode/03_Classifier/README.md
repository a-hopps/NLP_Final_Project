## Replicating the Paper's Main Experiments

To replicate the experiments, we recommend using the instructions provided in the notebook located in the `__ColabEnvironment` directory.

If you prefer to use a local machine, you need to install the required dependencies listed in `requirements.txt` (standard requirements from Google Colab plus the `scikit-learn` library).

## Replicating the LLM Model Experiments

To replicate the LLM-based experiments (e.g., DeepSeek-LLM, Llama-3, Gemma-3), follow these steps:

1. **Clone LLaMA Factory**

   The LLM experiments use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for training and evaluation. Clone the repository:

   ```sh
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   ```

   Install the required dependencies as described in the LLaMA Factory documentation.

2. **Use Provided Configuration**

   All configuration files for our LLM experiments are available in the [`LLaMA-Factory-Config`](../LLaMA-Factory-Config) directory at the root of this repository. These configs specify model parameters, and training settings.

   - Copy the relevant configuration file(s) from [`LLaMA-Factory-Config`](../LLaMA-Factory-Config) into your LLaMA Factory `configs` directory.
   - Update dataset paths in the config if your data is stored in a different location.

3. **Colab Environment**

    We used the ColabEnvironment_LLaMA_Factory.ipynb notebook from the __ColabEnvironment directory to work with the LLaMA Factory GUI in Google Colab. You can use this notebook to easily set up and run your own experiments in a Colab environment.
    Evaluation was performed using the integration code from ColabEnvironment_LLaMA_Factory notebook.
    In order to replicate the experiments you should also have a huggingface token available.

4. **Datasets Used**

   Training and evaluation were performed using the datasets from 01_TASK_DATASETS\03_Task_Datasets\03_DATASETS_ALPACA_ENCODING\OUR_DATASET

## Replicating the Experiments using Gpt4o, Gpt4o-mini, and o1-mini, o3-mini

To replicate the experiments with Gpt4o, Gpt4o-mini, and o1-mini, o3-mini use the notebooks from the `app_src/gpt_models_experiments` directory.

- `00_GPT_4_Experiments.ipynb` contains the experiments with Gpt4o and Gpt4o-mini.
- `00_GPT_O_Experiments.ipynb` contains the experiments with o1-mini, o3-mini.
- `00_GPT_O_Experiments 2025` contains the experiments with o3-mini on 2025 dataset.

Install the requirements from `app_src/gpt_models_experiments/requirements.txt`.

Results of the experiments from the paper are in `GPT_MODELS_RESULTS.xlsx`.

## Hardware Requirements

The experiments were conducted using a Colab environment with an A100 GPU (40GB) and a Linux distribution.

