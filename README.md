# bRAG AI: Retrieval-Augmented Fine-Tuning for Code Language Models


### Overview

**[bRAG AI](https://bragai.tech)** is an innovative Retrieval-Augmented Generation (RAG) framework designed to enhance code-centric language models through parameter-efficient fine-tuning and dynamic context retrieval. By leveraging advanced techniques like Low-Rank Adaptation (LoRA) and Fill-In-The-Middle (FIM) training, bRAG AI significantly improves domain-specific code generation and understanding.

## Video Demo

<video width="800" controls>
  <source src="assets/bRAGAI_Final_Video.mov" type="video/quicktime">
  Your browser does not support the video tag.
</video>

## Key Features
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

```bash
.
└── src                                                   
    ├── README.md                                         
    ├── eval                                              
    │   ├── code-eval                                     
    │   │   ├── codellama-base-eval.py                    
    │   │   ├── codellama-bragai-eval.py                  
    │   │   ├── humaneval/                                
    │   │   └── requirements.txt                          
    │   ├── rag.py                                        
    │   ├── requirements.txt                              
    │   └── zeroshot.py                                   
    └── finetune                                          
        ├── data                                          
        │   ├── README.md                                 
        │   ├── clone_gh_repos.py                         
        │   ├── prepare_dataset.py                        
        │   ├── push_to_hub.py                            
        │   └── requirements.txt                          
        ├── inference                                     
        └── training                                      
            ├── fim.py                                    
            ├── requirements.txt                          
            ├── run_peft.sh                               
            └── train.py                                  

13 directories, 42 files
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