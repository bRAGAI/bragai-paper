import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Load the Fine-Tuned LLM
model_name = "hababou/codellama-bragai" 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    revision="main"
)

if not hasattr(model, "hf_device_map"):
    model.cuda()

# Load Dataset for Fine-Tuning/Evaluation
dataset = load_dataset("hababou/hf-codegen-v2", split="train")  # Use your dataset
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Function for Prompt Generation
def generate_prompt(context, question):
    """
    Generate dynamic RAG-style prompts based on the context and the question.
    """
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    return prompt

# Function for Code Completion (or Response Generation)
def get_code_completion(context, question):
    """
    Generate code completions or responses using the fine-tuned model.
    """
    prompt = generate_prompt(context, question)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
context = """
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler =
"""
question = "How do I initialize the training setup with an accelerator?"
print("Generated Code/Response:\n", get_code_completion(context, question))

# Advanced Scenario: Custom Prompt for Retrieval Context
context_2 = """
# Implementation of hierarchical negative binomial regression with random effects.
import math
import torch
from scipy.optimize import minimize

# Function to calculate log-likelihood for the model
"""
question_2 = "How do I add random effects to this model?"
print("Generated Code/Response:\n", get_code_completion(context_2, question_2))

# Example of Evaluating Fine-Tuned Model
def evaluate_model(dataset, model, tokenizer):
    """
    Evaluate the fine-tuned model using custom prompts.
    """
    results = []
    for example in dataset:
        context = example["context"]
        question = example["question"]
        ground_truth = example["answer"]
        
        # Generate response
        generated = get_code_completion(context, question)
        
        # Compare with ground truth
        is_correct = ground_truth.strip().lower() in generated.strip().lower()
        results.append({"question": question, "generated": generated, "correct": is_correct})
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    return accuracy, results

# Fine-Tuning Setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the Model
trainer.train()

# Evaluate the Model
accuracy, evaluation_results = evaluate_model(dataset, model, tokenizer)
print(f"Model Accuracy: {accuracy:.2%}")

# Integrate Test Cases
def test_model_with_cases(test_cases):
    for idx, case in enumerate(test_cases, start=1):
        context = case["context"]
        question = case["question"]
        ground_truth = case["ground_truth"]

        generated = get_code_completion(context, question)
        is_correct = ground_truth.strip().lower() in generated.strip().lower()
        print(f"Test Case {idx}:")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Generated: {generated}")
        print(f"Correct: {is_correct}\n")

# Load Test Cases
test_cases = [
    {
        "context": """import torch
for epoch in range(epochs):
    for batch in dataloader:
""",
        "question": "How can I implement the training loop in PyTorch?",
        "ground_truth": "Include loss calculation, backward pass, and optimizer step inside the loop."
    },
    {
        "context": """import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = 
""",
        "question": "How do I add a dropout layer in a PyTorch model?",
        "ground_truth": "Use nn.Dropout() with a probability value."
    },
    {
        "context": """from sklearn.model_selection import train_test_split
data = ...
labels = ...
""",
        "question": "How can I split data into training and test sets?",
        "ground_truth": "Use train_test_split(data, labels, test_size=0.2) to split the data."
    },
    {
        "context": """import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x, y)
""",
        "question": "How do I add labels and a title to the plot?",
        "ground_truth": "Use plt.xlabel(), plt.ylabel(), and plt.title()."
    },
    {
        "context": """def preprocess_data(data):
    # Data cleaning
    return cleaned_data
""",
        "question": "How do I normalize the data in the preprocessing step?",
        "ground_truth": "Use MinMaxScaler or StandardScaler from sklearn."
    },
    {
        "context": """Hierarchical negative binomial model:
def negative_binomial_log_likelihood():
    pass
""",
        "question": "How do I include random effects in this model?",
        "ground_truth": "Introduce a term for random intercepts and estimate them."
    },
    {
        "context": """# Implementation of an LSTM-based sentiment analysis model
from torch.nn import LSTM, Embedding
class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
""",
        "question": "How do I add an LSTM layer to this model?",
        "ground_truth": "Use LSTM(input_size, hidden_size) and forward the embeddings."
    },
    {
        "context": """from langchain.vectorstores import Pinecone
pinecone.init(api_key='...')
vectorstore = 
""",
        "question": "How do I initialize a Pinecone vector store?",
        "ground_truth": "Use Pinecone.from_documents(documents, embedding_function)."
    },
]

# Run Test Cases
test_model_with_cases(test_cases)
