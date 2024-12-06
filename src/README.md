# bRAG AI: Evaluation and Fine-Tuning for Retrieval-Augmented Generation

This project explores the application of fine-tuned language models in Retrieval-Augmented Generation (RAG) workflows for code completion, statistical modeling, and domain-specific retrieval tasks. The research leverages hierarchical evaluation methods and fine-tuned causal language models to improve performance in real-world tasks, such as code understanding and answering research-driven queries.

## Key Features

1. **Fine-Tuned Models for Retrieval-Augmented Generation**:
   - The `hababou/codellama-bragai` fine-tuned model was specifically optimized for RAG tasks.
   - The model evaluates retrieval strategies and context-based completions in domains like programming and research.

2. **Dataset Integration**:
   - The project uses a curated dataset (`hababou/hf-codegen-v2`) featuring real-world problems and solutions for fine-tuning and evaluation.
   - The dataset `hababou/code-chat-assistant-v1`, combining LIMA and GUANACO, was utilized for training RAG chat capabilities, ensuring readiness for interactive use cases.
   - Non-code files were excluded during preprocessing to focus on programming-related content.

3. **Evaluation Metrics**:
   - Model performance is quantified using metrics like Pass@k, NDCG@10, and Recall@10.
   - Ground truth comparisons ensure the model maintains high relevance and accuracy in retrieval-augmented scenarios.

4. **Test Case Coverage**:
   - Includes 30 diverse scenarios, ranging from code generation to context-specific retrieval, ensuring broad applicability in various contexts.

## Fine-Tuning Process

### 1. **Parameter-Efficient Fine-Tuning**
   - **Low-Rank Adaptation (LoRA):**
     - LoRA decomposes weight update matrices into low-rank representations, enabling efficient fine-tuning without modifying the base model extensively.
   - **Quantization-Aware LoRA (QLoRA):**
     - Introduces 4-bit quantization to improve memory efficiency while preserving model quality.
   - **Fill-In-The-Middle (FIM) Training Objective:**
     - The FIM approach trains the model to predict missing segments within a given context, enhancing reasoning capabilities by leveraging bidirectional context.
   - **Training Configuration:**
     - Learning rate: `3 × 10^-4`
     - LoRA rank: `r = 32`, scaling factor: `α = 64`
     - Dropout: `0.1`
     - Maximum sequence length: `2048`
     - Hardware: NVIDIA A100 GPUs, 80GB VRAM

### 2. **Dataset Preparation**
   - The dataset for fine-tuning bRAG AI was curated from the top 25 repositories in the Hugging Face GitHub organization. These repositories contain high-quality code snippets, documentation, and issue discussions, providing a rich source of structured and unstructured knowledge for training.
   - Non-code files, such as configuration files and unrelated documentation, were excluded to ensure domain relevance. Tokenization was performed using Byte-Pair Encoding (BPE), optimized for programming languages.
   - The dataset, labeled `hababou/hf-codegen-v2` on Hugging Face, was structured for seamless integration with the fine-tuning process, including a 10% test split for performance evaluation.

### 3. **Training Methodology**
   - **Fill-In-The-Middle (FIM) Objective:**
     - Inspired by "Efficient Training of Language Models to Fill in the Middle," the FIM training objective enhances the model's ability to predict missing segments using both preceding and succeeding context. A 50% FIM rate was adopted, balancing standard left-to-right modeling with gap-filling tasks.
   - **Efficiency Metrics:**
     - Training was monitored using Weights & Biases (W&B), tracking metrics like training loss, evaluation loss, and learning rate schedules to ensure convergence.
   - **Hardware Setup:**
     - Fine-tuning was conducted on NVIDIA A100 GPUs with 80GB VRAM, employing gradient accumulation for memory efficiency.

## Code Completion Evaluation: Python HumanEval Results

To evaluate the model's capabilities post-fine-tuning, the Python HumanEval benchmark was used:

| Model                         | Pass@1 (%) |
|-------------------------------|------------|
| CodeLlama-7B-Instruct-hf (Base) | 34.10       |
| bRAG-AI-LoRA (Fine-Tuned)      | 33.95       |

*Table 3: Python HumanEval Pass@1 Results for Base and Fine-Tuned Models*

The negligible difference in scores demonstrates that fine-tuning preserves the foundational capabilities of the base model while adapting it for RAG tasks.

## Retrieval-Augmented Inference

The bRAG AI framework integrates a dynamic retrieval pipeline to fetch relevant external knowledge, such as:
   - GitHub repositories
   - Academic papers
   - Multimedia transcripts

The retrieved context is appended to the input, enabling the model to generate responses grounded in real-time information.

## Code Overview

```
.
├── README.md                   # Documentation
├── eval
│   ├── code-eval
│   │   ├── codellama-base-eval.py  # Evaluation script for base model
│   │   ├── codellama-bragai-eval.py  # Evaluation script for fine-tuned model
│   ├── humaneval/              # HumanEval benchmark integration
│   └── rag.py                  # Retrieval-augmented generation evaluation script
├── finetune
│   ├── data
│   │   ├── README.md               # Dataset preparation instructions
│   │   ├── clone_gh_repos.py       # Script to clone GitHub repositories
│   │   ├── prepare_dataset.py      # Script to prepare dataset for fine-tuning
│   │   ├── push_to_hub.py          # Script to upload datasets to Hugging Face Hub
│   └── training
│       ├── fim.py                  # Implements FIM (Fill-in-the-Middle) techniques
│       ├── train.py                # Fine-tuning script
```

## How to Run the Evaluation

1. **Setup the Environment:**
   Install the dependencies for evaluation:
   ```bash
   pip install -r eval/requirements.txt
   ```

2. **Run RAG Evaluation:**
   Use the `rag.py` script to evaluate the fine-tuned model:
   ```bash
   python eval/rag.py
   ```

3. **Run Code Evaluation:**
   Evaluate both base and fine-tuned models using the HumanEval benchmark:
   ```bash
   python eval/code-eval/codellama-base-eval.py
   python eval/code-eval/codellama-bragai-eval.py
   ```

4. **Analyze the Results:**
   The scripts output the accuracy and relevance metrics for test cases, including Pass@1 and qualitative insights.

## Performance Highlights

The project incorporates Retrieval-Augmented Generation into a dynamic workflow, demonstrating:

1. **Improved Retrieval Accuracy:**
   - The fine-tuned model shows high relevance in retrieving documents for code-centric tasks.

2. **Effective Code Completions:**
   - Generates syntactically and semantically accurate Python code snippets.

3. **Qualitative and Quantitative Insights:**
   - Example outputs illustrate the effectiveness of the RAG pipeline in solving both general and domain-specific tasks.

## Research Contributions

This project builds on state-of-the-art RAG techniques to:
   - Fine-tune causal language models for context-aware retrieval and generation.
   - Develop a benchmark framework for evaluating retrieval-augmented models.
   - Explore the intersection of generative AI and software engineering.

For detailed results and experimental setups, refer to the accompanying research paper.

## Future Work

1. **Integrating New Benchmarks:**
   - Expand to datasets like HumanEval and beyond for comprehensive evaluation.

2. **Combining Retrieval Sources:**
   - Enhance the retrieval mechanism to combine diverse knowledge bases dynamically.

3. **Scaling the RAG Workflow:**
   - Explore multi-modal RAG for applications in fields like healthcare and education.

## References

1. [Hugging Face Documentation](https://huggingface.co/docs/)
2. [LangChain RAG Framework](https://docs.langchain.com/docs/)
3. [bRAG AI Dataset Repository](https://huggingface.co/datasets/hababou/hf-codegen-v2)
4. [Code Chat Assistant Dataset](https://huggingface.co/datasets/hababou/code-chat-assistant-v1)

For questions or collaborations, feel free to reach out!

