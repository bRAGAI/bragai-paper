# hf-codegen

A repository containing Python scripts for collating code content from the public repositories of `huggingface` on GitHub. 

Resultant dataset: https://huggingface.co/datasets/sayakpaul/hf-codegen-v2. 

## To generate the dataset 

Make sure you have at least 50 GB of disk space.

1. Clone the repo and change to the `data` directory. 
2. Install the Python dependencies: `pip install -r requirements.`
3. Run `python parallel_clone_repos.py` to locally clone the public repositories situated under the `huggingface` GitHub org. You'd need to set up `GH_ACCESS_TOKEN` as the env variable (it can be your GitHub personal access token).
4. Log in to your HF account: `huggingface-cli login`.
5. Prepare the dataset, serialize in feather files, and upload them to the Hugging Face Hub: `python prepare_dataset.py`.
6. To finally have the dataset compatible with 🤗 Datasets (helps with downstream training), run `python push_to_hub.py`.

💡 Note that Step 6 was run on a separate machine with lots of RAM (240 GB). Steps 5 - 6 could have been clubbed together had we used a more capable machine from the get-go. 

The final dataset can be found here: [sayakpaul/hf-codegen-v2](https://hf.co/datasets/sayakpaul/hf-codegen-v2).