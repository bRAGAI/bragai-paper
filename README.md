# bRAG AI: Retrieval-Augmented Fine-Tuning for Code Language Models

### Overview

**[bRAG AI](https://bragai.tech)** is an innovative Retrieval-Augmented Generation (RAG) framework designed to enhance code-centric language models through parameter-efficient fine-tuning and dynamic context retrieval. By leveraging advanced techniques like Low-Rank Adaptation (LoRA) and Fill-In-The-Middle (FIM) training, bRAG AI significantly improves domain-specific code generation and understanding.

## Video Demo

<video width="800" controls>
  <source src="https://bragai.s3.us-east-1.amazonaws.com/bRAGAI+Final+Video.mp4" type="video/mp4">
  Your browser does not support the video tag. [Watch the Demo Video](https://bragai.s3.us-east-1.amazonaws.com/bRAGAI+Final+Video.mp4)
</video>

### Key Features
- **Parameter-Efficient Domain Adaptation:** Utilize LoRA and QLoRA for cost-effective model fine-tuning
- **Dynamic Multi-Source Retrieval:** Integrate context from GitHub repositories, academic papers, and multimedia transcripts
- **Enhanced Contextual Reasoning:** Improve code generation accuracy and domain adaptability
- **Optimized for Software Engineering:** Specifically designed for evolving code ecosystems

## Installation

```bash
# Clone the repository
git clone https://github.com/bRAGAI/bragai

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

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

## Research Paper
For a detailed techincal overview, please refer to our research paper:

- **Title:** bRAGAI: Retrieval-Augmented Fine-Tuning for Code LLMs
- **Author:** Taha H. Ababou
- **Affiliation:** Boston University
 
[Download Research Paper (PDF)](./bRAGAI_Final_Paper.pdf)

## Citation

If you use bRAGAI in your research, please cite:

```bibtex
@article{ababou2024bragAI,
  title={BRAG AI: Retrieval-Augmented Fine-Tuning for Code LLMs},
  author={Ababou, Taha H.},
  year={2024},
  institution={Boston University}
}
```