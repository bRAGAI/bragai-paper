# Cloning repositories from Huggingface's GitHub organization

This repository provides Python scripts for extracting and compiling code data from public repositories hosted by `huggingface` on GitHub.

## Steps to Generate the Dataset

Before starting, ensure your system has at least 50 GB of available storage.

1. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Use the following script to clone public repositories under the `huggingface` GitHub organization:
   ```bash
   python clone_gh_repos.py
   ```
   Make sure to configure the `GH_ACCESS_TOKEN` environment variable with your GitHub personal access token.

3. Authenticate with your Hugging Face account:
   ```bash
   huggingface-cli login
   ```

4. Generate the dataset, save it as Feather files, and upload it to the Hugging Face Hub:
   ```bash
   python prepare_dataset.py
   ```

5. Format the dataset for compatibility with Datasets, making it ready for downstream applications:
   ```bash
   python push_to_hub.py
   ```

Final Dataset Link: [hababou/hf-codegen-v2](https://huggingface.co/datasets/hababou/hf-codegen-v2)