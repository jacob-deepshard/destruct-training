# Destruct Training

## Overview

This project implements a training pipeline for an autoregressive language model using Proximal Policy Optimization (PPO) with KL divergence constraints and an autocurricular learning strategy based on difficulty estimation with a rolling frozen model.

### Key Features

- **PPO-based Training with KL Divergence Constraints**: Ensures stable training by balancing exploration and exploitation while constraining updates to stay within a trust region defined by KL divergence.
  
- **Autocurricular Learning Using Difficulty Estimation**: Dynamically adjusts the sampling weights of multiple datasets based on their estimated difficulty, promoting efficient learning by focusing on data that is optimally challenging for the model at its current state.

- **Multi-Dataset Training with Dynamic Sampling**: Trains the model on a diverse set of datasets, allowing it to generalize across various domains and styles.

- **Value Network for Advantage Estimation**: A separate value network estimates state values, which are used to compute advantages for PPO updates.

- **Model Checkpointing and Evaluation**: Periodically saves model checkpoints and evaluates performance on a validation set and a set of custom test prompts.

## Algorithm Details

### 1. Data Loading and Preprocessing

- **Datasets**: Loads multiple datasets specified in `config.yaml`, including Wikitext-103, BookCorpusOpen, CNN/DailyMail, and others.

- **Tokenization**: Uses a shared tokenizer (LlamaTokenizer) for all datasets. Special handling is implemented for datasets with unique structures:
  - **DailyDialog**: Concatenates dialog turns into a single string.
  - **EuroParl**: Extracts English text from bilingual translations.

### 2. Difficulty Estimation

- **Frozen Model**: A static copy of the current model is maintained (frozen) to estimate the difficulty of each dataset without being influenced by recent training updates.

- **Process**:
  - For each dataset:
    - Samples a small subset (e.g., 100 samples).
    - Computes the loss using the frozen model.
    - Calculates the mean loss (difficulty) and variance for the dataset.
  - Logs these metrics to TensorBoard for monitoring.

### 3. Dynamic Sampling Weight Update

- **Sampling Weights**: Initialized equally for all datasets.

- **Update Mechanism**:
  - Computes a combined metric using the difficulties and variances.
  - Calculates the gradient of this combined metric.
  - Updates the sampling weights using a second-order dynamics approach:
    ```python
    sampling_weights += eta * gradient
    sampling_weights = np.clip(sampling_weights, a_min=0, a_max=None)
    sampling_weights /= np.sum(sampling_weights)
    ```
  - This approach increases the probability of sampling from datasets that are currently more informative for the model.

### 4. Training with PPO

- **Policy Network**: The main language model (LlamaForCausalLM).

- **Value Network**: A separate neural network that estimates the value of each state.

- **Training Loop**:
  - **Batch Sampling**: Batches are sampled from the datasets according to the updated sampling weights.
  - **Forward Pass**: The model processes the input to generate outputs and hidden states.
  - **Reward Computation**: A custom reward function computes rewards for each sample (e.g., penalizing repetitions).
  - **Advantage Estimation**: Calculates advantages using Generalized Advantage Estimation (GAE):
    ```python
    returns, advantages = compute_gae(rewards, old_values, gamma, lam)
    ```
  - **PPO Updates**:
    - Performs multiple PPO epochs.
    - Calculates the surrogate loss and applies clipping to ensure updates stay within the trust region.
    - Introduces a KL divergence constraint to prevent drastic policy changes.
    - Optimizes the policy and value networks using gradient descent with gradient accumulation and mixed precision training.

### 5. Model Updates and Checkpointing

- **Frozen Model Update**: Periodically updates the frozen model with the current model's weights to keep difficulty estimation relevant.

- **Checkpointing**:
  - Saves model checkpoints at the end of each epoch.
  - If validation is performed, saves the best model based on validation loss.

### 6. Evaluation

- **Validation Loop**: Computes validation loss on a separate dataset to monitor overfitting and generalization.

- **Custom Test Prompts**: Generates text for various prompts covering creative writing, technical content, code snippets, and multiple languages to qualitatively assess the model's performance.

## Usage

### Prerequisites

- **Environment**:
  - Python 3.7+
  - Access to GPU(s) for training (CUDA-enabled device recommended)

- **Python Packages**:
  - `torch`
  - `transformers`
  - `datasets`
  - `numpy`
  - `tqdm`
  - `pyyaml`

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/destruct_training.git
   cd destruct_training
   ```

2. **Install Dependencies**:

   Using Poetry:

   ```bash
   poetry install
   ```

   Or using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- **Edit `config.yaml`**:

  Adjust the hyperparameters, dataset configurations, and other settings as needed. For example:

  ```yaml
  learning_rate: 1e-5
  num_epochs: 10
  batch_size: 8
  # ... other configurations ...
  ```

### Running the Training Script

```bash
poetry run python -m destruct_training --config config.yaml
```

### Monitoring Training

- **TensorBoard**:

  Start TensorBoard to visualize training metrics:

  ```bash
  tensorboard --logdir=./logs
  ```

- **Access**:

  Open your web browser and navigate to `http://localhost:6006/` to monitor training progress.

### Notes

- Ensure that you have access to the specified pre-trained model (`meta-llama/Llama-3.2-1B`) or replace it with an available model.

- Adjust `difficulty_sample_size` in `config.yaml` if you encounter memory issues during difficulty estimation.

## License

Copyright 2024 Jacob Valdez & Dylan Lanigan. See the [LICENSE](LICENSE) file for details.
