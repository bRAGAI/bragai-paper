# bRAG AI: Pioneering Retrieval-Augmented Fine-Tuning for Code LLMs

### Overview

**[bRAG AI](https://bragai.tech)** redefines the boundaries of code-centric language models by blending cutting-edge Retrieval-Augmented Generation (RAG) techniques with parameter-efficient fine-tuning. Tailored for dynamic and evolving software ecosystems, bRAG AI leverages advanced strategies such as Low-Rank Adaptation (LoRA) and Fill-In-The-Middle (FIM) training to deliver unmatched domain-specific adaptability and precision.

## Video Demo

[Experience bRAG AI in Action](https://bragai.s3.us-east-1.amazonaws.com/bRAGAI+Final+Video.mp4)

## Key Features

- **Efficient Fine-Tuning:** Harness the power of LoRA and QLoRA to fine-tune models with minimal computational overhead.
- **Dynamic Context Retrieval:** Seamlessly integrate knowledge from diverse sources like GitHub repositories, academic papers, and multimedia transcripts.
- **Context-Aware Reasoning:** Achieve exceptional accuracy in code generation and domain-specific adaptability through real-time context enrichment.
- **Engineered for Evolution:** Purpose-built to address the challenges of ever-changing codebases and frameworks.

## Installation

Get started with bRAG AI in a few simple steps:

```bash
# Clone the repository
git clone https://github.com/bRAGAI/bragai

# Navigate to the project directory
cd bragai

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
│       ├── fim.py                  # Implements FIM (Fill-in-The-Middle) techniques
│       ├── requirements.txt        # Dependencies for training
│       ├── run_peft.sh             # Shell script for PEFT training
│       └── train.py                # Fine-tuning script
```

## Research Paper

Explore the technical foundations of bRAG AI in our detailed research paper:

- **Title:** bRAG AI: Retrieval-Augmented Fine-Tuning for Code LLMs
- **Author:** Taha H. Ababou
- **Affiliation:** Boston University

[Download the Research Paper (PDF)](./docs/bRAGAI_Final_Paper.pdf)

## How to Cite

If bRAG AI contributes to your research, please use the following citation:

```bibtex
@article{ababou2024bragAI,
  title={bRAG AI: Retrieval-Augmented Fine-Tuning for Code LLMs},
  author={Ababou, Taha H.},
  year={2024},
  institution={Boston University}
}
```

## Future Directions

bRAG AI represents a significant leap forward, but we’re just getting started. Upcoming enhancements include:

- **Dynamic Retrieval Optimization:** Implement smarter, adaptive retrievers to further refine performance.
- **Multi-Modal Applications:** Extend RAG capabilities to support fields like healthcare, finance, and education.
- **Integrated Fine-Tuning:** Develop synchronized fine-tuning for both retrieval systems and LLMs to improve synergy.

## Join Us

We’re on a mission to revolutionize code LLMs. If you’re interested in contributing, collaborating, or simply learning more, we’d love to hear from you. Reach out via [our website](https://bragai.tech) or explore the repository today!

---

**bRAG AI: Empowering Developers, One Line of Code at a Time.**

