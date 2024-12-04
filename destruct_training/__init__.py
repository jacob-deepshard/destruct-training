#!/usr/bin/env python
# coding: utf-8

"""
Train/Test Research Script for Autoregressive Language Model with PPO and Autocurricular Learning

This script implements a training pipeline for a language model using Proximal Policy Optimization (PPO)
with KL divergence constraints and an autocurricular learning strategy based on difficulty estimation
using a rolling frozen model.

Key Features:
- PPO-based training with KL divergence constraints
- Autocurricular learning using difficulty estimation
- Multi-dataset training with dynamic sampling
- Value network for advantage estimation
- Model checkpointing and evaluation

Assumptions:
- A pre-trained Llama-3.2-1B model is available via HuggingFace Transformers.
- Access to the underlying PyTorch layers is available.
- The environment has sufficient computational resources for training.

Dependencies:
- torch
- transformers
- numpy
- datasets
- tqdm

Usage:
1. Install package and dependencies:
   poetry install

2. Set up environment variables (optional):
   export TRANSFORMERS_CACHE=/path/to/cache

3. Run training:
   poetry run python -m destruct_training --config config.yaml

4. Monitor outputs:
   - Training metrics will be printed to stdout and logged to TensorBoard
   - Model checkpoints saved to ./model_checkpoints/
   - Validation metrics will be logged
   - Test results shown after training

Note: Requires access to meta-llama/Llama-3.2-1B model.
Contact Meta AI for model access and license.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import yaml
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Set up argument parsing for configuration
parser = argparse.ArgumentParser(description='Train a Language Model with PPO and Autocurricular Learning')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Set random seeds for reproducibility
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU, which may be slow.")

# Enable mixed precision training
use_amp = config.get('use_amp', False)
scaler = torch.amp.GradScaler(enabled=use_amp)

# Load the pre-trained Llama model and tokenizer
model_name = config['model_name']  # Placeholder name; replace with actual model if available
print(f'Loading model: {model_name}')
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.model_max_length = config.get('max_seq_length', 512)
# model = LlamaForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model.to(device)
model.train()

# Initialize optimizer for policy network
learning_rate = float(config['learning_rate'])
print(f'Learning rate: {learning_rate}')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize learning rate scheduler
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Create a frozen copy of the model for difficulty estimation
frozen_model = LlamaForCausalLM.from_pretrained(model_name)
frozen_model.to(device)
frozen_model.eval()  # Set to evaluation mode

# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(ValueNetwork, self).__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # We take the hidden state corresponding to the last token
        last_hidden_state = hidden_states[:, -1, :]
        value = self.value_head(last_hidden_state).squeeze(-1)
        return value

value_network = ValueNetwork(model.config.hidden_size).to(device)
value_optimizer = optim.AdamW(value_network.parameters(), lr=learning_rate)

# Hyperparameters
num_epochs = config['num_epochs']
batch_size = config['batch_size']
ppo_epochs = config['ppo_epochs']
gamma = config.get('gamma', 0.99)
lam = config.get('lambda', 0.95)  # GAE lambda
epsilon = config.get('epsilon', 0.1)  # PPO clipping parameter
kl_target = config.get('kl_target', 0.01)  # KL divergence target
eta = config.get('eta', 0.1)  # Learning rate for sampling weights
freeze_interval = config.get('freeze_interval', 2)  # Epochs between updating the frozen model
grad_accumulation_steps = config.get('grad_accumulation_steps', 1)
max_seq_length = config.get('max_seq_length', 512)

# Load data subsets as HuggingFace datasets and tokenize
data_subset_infos = config['data_subsets']
data_subsets = []
text_column_mapping = config['text_column_mapping']

print("Loading and tokenizing datasets...")
for i, subset_info in enumerate(data_subset_infos):
    dataset_name = subset_info['dataset_name']
    subset_name = subset_info.get('subset_name', None)
    split = subset_info.get('split', 'train')
    dataset = load_dataset(dataset_name, subset_name, split=split)
    text_column = text_column_mapping[str(i)]

    # Preprocess and tokenize the dataset
    def tokenize_function(examples):
        if i == 8:
            # For DailyDialog, concatenate dialog turns
            texts = [' '.join(dialog) for dialog in examples[text_column]]
        elif i == 9:
            # For EuroParl, extract English translations
            texts = [translation['en'] for translation in examples[text_column]]
        else:
            texts = examples[text_column]
        return tokenizer(texts, truncation=True, max_length=max_seq_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_subsets.append(tokenized_dataset)

# Initialize sampling weights uniformly
sampling_weights = np.ones(len(data_subsets)) / len(data_subsets)

# Preparing the evaluation metric

# Define the evaluation function
def evaluate_model_output(prompt, generated_output, expected_output=None, context=None):
    test_case = LLMTestCase(
        input=prompt,
        actual_output=generated_output,
        expected_output=expected_output,
        context=context
    )
    # Instantiate the desired evaluation metric
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    # Measure the test case
    answer_relevancy_metric.measure(test_case)
    # Log or print the evaluation results
    score = answer_relevancy_metric.score
    reason = answer_relevancy_metric.reason
    print(f"Evaluation Score: {score}, Reason: {reason}")
    return score, reason


# Define the custom reward function
def compute_rewards(input_ids, model_outputs):
    """
    Compute rewards for the given inputs and model outputs.
    For example, penalize repetitions or undesired outputs.
    """
    # Placeholder implementation
    rewards = torch.zeros(input_ids.size(0)).to(device)
    # Example: Negative reward for repeating the same token consecutively
    for i in range(input_ids.size(0)):
        tokens = input_ids[i]
        num_repeats = sum([1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx-1]])
        rewards[i] = -num_repeats * 0.1  # Penalize repetitions
    return rewards

# Define Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, gamma, lam):
    """
    Compute returns and advantages using GAE.
    """
    advantages = torch.zeros_like(rewards).to(device)
    returns = torch.zeros_like(rewards).to(device)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return returns, advantages

# Function to sample batches according to updated weights
def sample_batches(data_subsets, sampling_weights, batch_size):
    batches = []
    num_batches = len(data_subsets)
    subset_indices = np.random.choice(len(data_subsets), size=num_batches, p=sampling_weights)
    for idx in subset_indices:
        dataset = data_subsets[idx]
        # Random sampling from the tokenized dataset
        batch = dataset.shuffle(seed=random.randint(0, 1e6)).select(range(batch_size))
        batches.append((batch, idx))
    return batches

# Initialize TensorBoard writer
log_dir = config.get('log_dir', "./logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Global step counter for TensorBoard
global_step = 0
best_val_loss = float('inf')

# Prepare validation data loader
val_dataset = None
if config.get('validation_dataset'):
    val_info = config['validation_dataset']
    val_dataset_raw = load_dataset(val_info['dataset_name'], val_info.get('subset_name', None), split=val_info.get('split', 'validation'))
    val_text_column = val_info['text_column']
    def tokenize_function(examples):
        texts = examples[val_text_column]
        return tokenizer(texts, truncation=True, max_length=max_seq_length)
    val_dataset = val_dataset_raw.map(tokenize_function, batched=True, remove_columns=val_dataset_raw.column_names)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Main training loop
print("Starting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    # Step 1: Estimate difficulty of each data subset using the frozen model
    difficulties = []
    variances = []

    for i, dataset in enumerate(data_subsets):
        losses = []
        # Sample a small subset for difficulty estimation
        sample_size = config.get('difficulty_sample_size', 100)
        dataset_sample = dataset.shuffle(seed=seed).select(range(sample_size))
        data_loader = DataLoader(dataset_sample, batch_size=batch_size, num_workers=4)

        for batch_data in data_loader:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)

            with torch.no_grad():
                outputs = frozen_model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss.item()
                losses.append(loss)

        mu_i = np.mean(losses)
        sigma_i = np.var(losses)
        difficulties.append(mu_i)
        variances.append(sigma_i)

        # Log the difficulty and variance
        writer.add_scalar(f'Difficulty/Subset_{i}', mu_i, epoch)
        writer.add_scalar(f'Variance/Subset_{i}', sigma_i, epoch)

    # Step 2: Update sampling weights using second-order dynamics
    difficulties = np.array(difficulties)
    variances = np.array(variances)
    combined_metric = difficulties + variances
    gradient = np.gradient(combined_metric)
    sampling_weights += eta * gradient
    sampling_weights = np.clip(sampling_weights, a_min=0, a_max=None)
    sampling_weights /= np.sum(sampling_weights)  # Normalize

    print(f"Sampling weights: {sampling_weights}")

    # Log sampling weights
    for idx, weight in enumerate(sampling_weights):
        writer.add_scalar(f'SamplingWeight/Subset_{idx}', weight, epoch)

    # Step 3: Sample batches according to updated weights
    sampled_batches = sample_batches(data_subsets, sampling_weights, batch_size)

    # Step 4: Iterate over sampled batches
    for batch_idx, (batch_data, subset_idx) in enumerate(tqdm(sampled_batches, desc="Training Batches")):
        try:
            # Prepare inputs
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)

            batch_size_actual = input_ids.size(0)

            # Get old policy outputs (needed for PPO)
            with torch.no_grad():
                old_outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                old_logits = old_outputs.logits  # (batch_size, seq_len, vocab_size)
                old_log_probs = nn.functional.log_softmax(old_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                old_values = value_network(old_outputs.hidden_states[-1]).detach()  # (batch_size)

            # Compute rewards
            rewards = compute_rewards(input_ids, old_outputs)  # (batch_size)

            # Log rewards
            writer.add_scalar('Reward/Mean', rewards.mean().item(), global_step)
            writer.add_scalar('Reward/Std', rewards.std().item(), global_step)

            # Append a zero to values to match rewards length after shifting
            old_values = torch.cat((old_values, torch.tensor([0.0]).to(device)), dim=0)

            # Compute returns and advantages
            returns, advantages = compute_gae(rewards, old_values, gamma, lam)
            advantages = advantages.detach()
            returns = returns.detach()

            # PPO Updates
            for ppo_epoch in range(ppo_epochs):
                # Get new policy outputs
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                    log_probs = nn.functional.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

                # Flatten the tensors for simplicity
                old_log_probs_flat = old_log_probs.view(-1, old_log_probs.size(-1))
                log_probs_flat = log_probs.view(-1, log_probs.size(-1))
                input_ids_flat = input_ids.view(-1)

                # Gather log probabilities of the actions taken
                old_log_probs_action = old_log_probs_flat[range(old_log_probs_flat.size(0)), input_ids_flat]
                log_probs_action = log_probs_flat[range(log_probs_flat.size(0)), input_ids_flat]

                # Reshape back to (batch_size, seq_len)
                old_log_probs_action = old_log_probs_action.view(batch_size_actual, -1)
                log_probs_action = log_probs_action.view(batch_size_actual, -1)

                # Calculate ratios
                ratios = torch.exp(log_probs_action - old_log_probs_action)  # (batch_size, seq_len)

                # Use the last token for ratios and advantages
                ratios = ratios[:, -1]  # (batch_size)
                advantages_batch = advantages[:batch_size_actual]  # Adjust the size

                # Calculate surrogate loss
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL divergence constraint
                kl_div = nn.functional.kl_div(log_probs_action, old_log_probs_action, reduction='batchmean')
                if kl_div.item() > kl_target * 1.5:
                    print(f"Early stopping PPO updates due to high KL divergence: {kl_div.item()}")
                    break

                # Update policy
                optimizer.zero_grad()
                scaler.scale(policy_loss).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Update value network
                new_values = value_network(outputs.hidden_states[-1])
                value_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns[:batch_size_actual])
                value_optimizer.zero_grad()
                scaler.scale(value_loss).backward()
                torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
                scaler.step(value_optimizer)
                scaler.update()

                # Log losses and KL divergence
                writer.add_scalar('Loss/Policy', policy_loss.item(), global_step)
                writer.add_scalar('Loss/Value', value_loss.item(), global_step)
                writer.add_scalar('KL_Divergence', kl_div.item(), global_step)

                global_step += 1  # Increment global step

            # Gradient accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                value_optimizer.step()
                optimizer.zero_grad()
                value_optimizer.zero_grad()

        except Exception as e:
            print(f"Error during training batch {batch_idx}: {e}")
            continue

    # Step 5: Update the frozen model periodically
    if (epoch + 1) % freeze_interval == 0:
        frozen_model.load_state_dict(model.state_dict())
        frozen_model.eval()
        print("Updated the frozen model for difficulty estimation.")
        # Log the model update
        writer.add_scalar('FrozenModelUpdate', epoch + 1, epoch)

    # Step 6: Validation loop with evaluation
    if val_dataset is not None:
        model.eval()
        val_losses = []
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation Batches"):
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                
                # Generate outputs
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_seq_length,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
                
                # Decode inputs and outputs
                prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Evaluate each generated text
                for prompt, generated_text in zip(prompts, generated_texts):
                    score, reason = evaluate_model_output(prompt, generated_text)
                    # Log the score to TensorBoard
                    writer.add_scalar('Evaluation/AnswerRelevancy', score, global_step)
                    global_step += 1  # Increment global step if needed
                
                # Optional: Compute loss if you're also evaluating against ground truth
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = config.get('checkpoint_dir', "./model_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
            writer.add_text('Checkpoint', f"Epoch {epoch+1} best model saved.", epoch)
    else:
        # Optional: Save the model checkpoint
        checkpoint_dir = config.get('checkpoint_dir', "./model_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model checkpoint at {model_save_path}")
        # Log model checkpoint saving
        writer.add_text('Checkpoint', f"Epoch {epoch+1} model saved.", epoch)

    # Step the learning rate scheduler
    lr_scheduler.step()

# Close the TensorBoard writer
writer.close()

print("Training completed.")

# Test the model
def test_model(test_texts):
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Testing"):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_seq_length).to(device)
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    return generated_texts

# Example test cases covering different domains and styles
test_samples = [
    # Creative writing prompts
    "Once upon a time in a land far away,",
    "The dark clouds gathered ominously as",
    "She opened the mysterious letter and read:",
    
    # Philosophical questions
    "The purpose of life is",
    "The relationship between consciousness and reality can be described as",
    "Free will versus determinism:",
    
    # Technical writing
    "The key advantages of neural networks include",
    "To implement a binary search algorithm,",
    "The difference between HTTP and HTTPS is",
    
    # Analytical prompts
    "The main causes of climate change are",
    "Economic inequality can be addressed by",
    "The role of artificial intelligence in modern society",
    
    # Conversational prompts
    "Hey, what do you think about",
    "Could you explain to me how",
    "I've been wondering lately if",
    
    # Task-oriented prompts
    "Write a recipe for chocolate cake:",
    "Steps to troubleshoot a computer that won't start:",
    "Tips for effective time management include",
    
    # Academic writing
    "The thesis of this research paper argues that",
    "In conclusion, the evidence suggests",
    "According to recent studies in neuroscience,"
    
    # Code snippets
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "async function getData() {",
    
    # Synthetic/constructed languages
    "mi toki e ni: toki pona li pona",  # Toki Pona
    "-.-- --- ..- / .- .-. . / .... . .-. .",  # Morse code
    "01001000 01100101 01101100 01101100 01101111",  # Binary
    
    # Mathematical notation
    "∀x ∈ ℝ, ∃y ∈ ℕ such that",
    "∫₀^∞ e^(-x²) dx =",
    "lim_{x→∞} (1 + 1/x)^x =",
    
    # Multi-language prompts
    "Buenos días, ¿cómo estás?", # Spanish
    "こんにちは、元気ですか？", # Japanese
    "Здравствуйте, как дела?", # Russian
    
    # Signal processing
    "~^~_~^~_~^~_", # Sine wave ASCII art
    "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁", # Audio waveform
    "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", # Braille loading animation
    
    # Esoteric programming
    "+++++ +++[- >++++ ++++< ]>+++ .", # Brainfuck
    ":dup :bernoulli if bye", # Forth
    "(λx.xx)(λx.xx)", # Lambda calculus
    
    # Network/Protocol patterns  
    "GET /api/v1/users HTTP/1.1\nHost:",
    "ssh-rsa AAAAB3NzaC1yc2EA...",
    "ping 192.168.1.1 -c 4",
    
    # Chemical formulas
    "C₆H₁₂O₆ + 6O₂ →",
    "Na⁺ + Cl⁻ →",
    "CH₃COOH ⇌ CH₃COO⁻ + H⁺"
]

generated_outputs = test_model(test_samples)
for prompt, output in zip(test_samples, generated_outputs):
    print(f"Prompt: {prompt}\nGenerated: {output}\n")
