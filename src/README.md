# bRAG AI: Evaluation and Fine-Tuning for Retrieval-Augmented Generation

This project explores the application of fine-tuned language models in Retrieval-Augmented Generation (RAG) workflows for code completion, statistical modeling, and domain-specific retrieval tasks. The research leverages hierarchical evaluation methods and fine-tuned causal language models to improve performance in real-world tasks, such as code understanding and answering research-driven queries.

## Key Features

1. **Fine-Tuned Models for Retrieval-Augmented Generation**:
   - The `hababou/codellama-bragai` fine-tuned model was specifically optimized for Retrieval-Augmented Generation tasks.
   - The model evaluates hierarchical negative binomial models, retrieval strategies, and context-based completions in domains like programming and research.

2. **Dataset Integration**:
   - The project uses a curated dataset (`hababou/hf-codegen-v2`) featuring real-world problems and solutions for fine-tuning and evaluation.
   - The dataset `hababou/code-chat-assistant-v1`, a mix of LIMA and GUANACO with proper formatting, was utilized for training the RAG chat capabilities, ensuring readiness for interactive use cases.
   - Test cases are designed to evaluate the system's ability to generate accurate responses for both retrieval-based and generative tasks.

3. **Evaluation Metrics**:
   - Model performance is quantified using metrics like Pass@k and NDCG@10.
   - Ground truth comparisons ensure the model maintains high relevance and accuracy in retrieval-augmented scenarios.

4. **Test Case Coverage**:
   - Includes 30 diverse scenarios, ranging from hierarchical modeling to code generation, ensuring broad applicability in various contexts.

## Code Completion Evaluation: Python HumanEval Results

To evaluate whether fine-tuning with QLoRA preserves the model’s foundational knowledge and avoids catastrophic forgetting, we ran the Python HumanEval benchmark on our fine-tuned bRAG AI model. This benchmark measures the functional correctness of generated Python programs, focusing on the Pass@1 metric, which evaluates the proportion of problems solved correctly on the first attempt.

| Model                         | Pass@1 (%) |
|-------------------------------|------------|
| CodeLlama-7B-Instruct-hf (Base) | 34.10       |
| bRAG-AI-LoRA (Fine-Tuned)      | 33.95       |

*Table 3: Python HumanEval Pass@1 Results for Base and Fine-Tuned Models*

The results, as presented in Table 3, indicate that the fine-tuned model achieves a Pass@1 score of 33.95%, marginally lower than the base model’s 34.10%. This negligible difference demonstrates that fine-tuning with LoRA preserves the base model’s general-purpose coding capabilities while enabling domain-specific adaptations. The baseline model’s performance aligns with the reported results for Code Llama models in the original paper, further validating the foundational model’s robustness. This result underscores the efficacy of parameter-efficient fine-tuning in maintaining foundational knowledge.

## Code Overview

The project structure includes the following key components:

```
.
├── README.md                   # Documentation
├── eval
│   ├── code-eval
│   │   ├── codellama-base-eval.py  # Evaluation script for base model
│   │   ├── codellama-bragai-eval.py  # Evaluation script for fine-tuned model
│   │   ├── humaneval/              # HumanEval benchmark integration
│   │   └── requirements.txt        # Dependencies for code evaluation
│   ├── rag.py                      # RAG evaluation script
│   ├── requirements.txt            # Dependencies for evaluation scripts
│   └── zeroshot.py                 # Zero-shot evaluation script
├── finetune
│   ├── data
│   │   ├── README.md               # Dataset preparation instructions
│   │   ├── clone_gh_repos.py       # Script to clone GitHub repositories
│   │   ├── prepare_dataset.py      # Script to prepare dataset for fine-tuning
│   │   ├── push_to_hub.py          # Script to upload datasets to Hugging Face Hub
│   │   └── requirements.txt        # Dependencies for dataset preparation
│   ├── inference/                  # Inference-related scripts
│   └── training
│       ├── fim.py                  # Implements FIM (Fill-in-the-Middle) techniques
│       ├── requirements.txt        # Dependencies for training
│       ├── run_peft.sh             # Shell script for PEFT training
│       └── train.py                # Fine-tuning script
```

## How to Run the Evaluation

1. **Setup the Environment**:
   Install the dependencies for evaluation:
   ```bash
   pip install -r eval/requirements.txt
   ```

2. **Run RAG Evaluation**:
   Use the `rag.py` script to evaluate the fine-tuned model:
   ```bash
   python eval/rag.py
   ```

3. **Run Code Evaluation**:
   Evaluate both base and fine-tuned models using HumanEval benchmark:
   ```bash
   python eval/code-eval/codellama-base-eval.py
   python eval/code-eval/codellama-bragai-eval.py
   ```

4. **Analyze the Results**:
   The scripts output the accuracy and relevance metrics for test cases, including Pass@1 and qualitative insights.

## Performance Highlights

The project incorporates Retrieval-Augmented Generation into a hierarchical modeling pipeline, demonstrating:

1. **Improved Retrieval Accuracy**:
   - The fine-tuned model shows high relevance in retrieving documents for hierarchical models.

2. **Effective Code Completions**:
   - Generates syntactically and semantically accurate Python code snippets.

3. **Qualitative and Quantitative Insights**:
   - Example outputs illustrate the effectiveness of the RAG pipeline in solving both general and domain-specific tasks.

## Research Contributions

This project builds on state-of-the-art RAG techniques to:
   - Fine-tune causal language models for context-aware retrieval and generation.
   - Develop a benchmark framework for hierarchical model evaluation.
   - Explore the intersection of generative AI and statistical modeling.

For detailed results and experimental setups, refer to the accompanying research paper.

## Future Work

1. **Integrating New Benchmarks**:
   - Expand to datasets like HumanEval and beyond for comprehensive evaluation.

2. **Combining Retrieval Sources**:
   - Enhance the retrieval mechanism to combine diverse knowledge bases dynamically.

3. **Scaling the RAG Workflow**:
   - Explore multi-modal RAG for applications in fields like healthcare and education.

## References

1. [Hugging Face Documentation](https://huggingface.co/docs/)
2. [LangChain RAG Framework](https://docs.langchain.com/docs/)
3. [bRAG AI Dataset Repository](https://huggingface.co/datasets/hababou/hf-codegen-v2)
4. [Code Chat Assistant Dataset](https://huggingface.co/datasets/hababou/code-chat-assistant-v1)

For questions or collaborations, feel free to reach out!

