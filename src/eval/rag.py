# RAG Implementation with OpenAI Embeddings and Hugging Face Models

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from openai.embeddings_utils import get_embedding  # OpenAI embedding utility
from sklearn.metrics import ndcg_score
import numpy as np
import openai

from dotenv import load_dotenv
load_dotenv()

#### SETUP OPENAI EMBEDDINGS ####
# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    """
    Generate embeddings using OpenAI's `text-embedding-3-small`.
    """
    response = openai.Embedding.create(input=text, model="text-embedding-3-small")
    return response["data"][0]["embedding"]

#### MODEL LOADING ####

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7B-Instruct-hf")
baseline_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7B-Instruct-hf")
finetuned_model = AutoModelForCausalLM.from_pretrained("hababou/codellama-bragai")

# Set up generation pipelines
baseline_pipeline = pipeline("text-generation", model=baseline_model, tokenizer=tokenizer, device=0)
finetuned_pipeline = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer, device=0)

#### INDEXING ####

def load_directory_chunks(directory_path):
    """
    Recursively load all text files from the given directory, split them into chunks, and return a list of documents.
    """
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".txt", ".py", ".md")):  # Adjust for specific file extensions
                file_path = os.path.join(root, file)
                
                # Load file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = text_splitter.split_text(content)
                
                # Create document objects
                all_docs.extend(chunks)
    
    return all_docs

#### LOAD DATASET ####

# Path to the large dataset (adjust to match `hf-codegen-v2`)
dataset_directory = "../finetune/data/hf_codegen_v2/data/train-00000-of-00001.parquet"

# Load and split all files in the dataset
docs = load_directory_chunks(dataset_directory)

# Embed and initialize Pinecone vector store
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
vectorstore = Pinecone.from_documents(
    documents=splits,
    embedding_function=get_openai_embedding,
    index_name="rag-index"
)

retriever = vectorstore.as_retriever()

#### RETRIEVAL AND GENERATION ####

# Define RAG pipeline
def generate_rag_response(question, retriever, generation_pipeline, k=5):
    """
    Generate a response using Retrieval-Augmented Generation (RAG).
    """
    # Retrieve top-k relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    top_k_context = "\n\n".join([doc.page_content for doc in retrieved_docs[:k]])
    
    # Generate response
    input_prompt = f"Answer the question based only on the following context:\n{top_k_context}\n\nQuestion: {question}"
    response = generation_pipeline(input_prompt, max_length=512, do_sample=False)
    
    return response[0]["generated_text"]

#### EVALUATION METRICS ####

def evaluate_rag_metrics(questions, ground_truths, retriever, generation_pipeline, k=5):
    """
    Evaluate RAG performance using Pass@k, NDCG@10, and Recall@10.
    """
    ndcg_scores = []
    recall_scores = []
    pass_at_k = []
    
    for question, ground_truth in zip(questions, ground_truths):
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        
        # Generate response using RAG
        response = generate_rag_response(question, retriever, generation_pipeline, k=k)
        
        # Relevance calculation
        relevance = [1 if ground_truth in text else 0 for text in retrieved_texts]
        
        # NDCG@10
        ndcg = ndcg_score([relevance], [relevance[:10]])
        ndcg_scores.append(ndcg)
        
        # Recall@10
        recall = sum(relevance[:10]) / sum(relevance) if sum(relevance) > 0 else 0
        recall_scores.append(recall)
        
        # Pass@k
        pass_at_k.append(1 if ground_truth in response else 0)
    
    metrics = {
        "Pass@k": np.mean(pass_at_k),
        "NDCG@10": np.mean(ndcg_scores),
        "Recall@10": np.mean(recall_scores)
    }
    return metrics

# Test data
questions = ["What is feature extraction in datasets?", "How do you preprocess data?"]
ground_truths = ["Feature extraction involves ...", "Preprocessing includes ..."]

# Evaluate RAG performance
rag_metrics = evaluate_rag_metrics(questions, ground_truths, retriever, finetuned_pipeline)
print("RAG Metrics:", rag_metrics)
