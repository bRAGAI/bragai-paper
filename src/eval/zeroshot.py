#### ZERO-SHOT EVALUATION ####

def generate_zero_shot_response(question, generation_pipeline):
    """
    Generate a response using the LLM without any retrieval context.
    """
    input_prompt = f"Answer the question to the best of your ability:\n\nQuestion: {question}"
    response = generation_pipeline(input_prompt, max_length=512, do_sample=False)
    return response[0]["generated_text"]

def evaluate_zero_shot_metrics(questions, ground_truths, generation_pipeline):
    """
    Evaluate zero-shot performance using Pass@k.
    """
    pass_at_1 = []
    
    for question, ground_truth in zip(questions, ground_truths):
        # Generate response
        response = generate_zero_shot_response(question, generation_pipeline)
        pass_at_1.append(1 if ground_truth in response else 0)
    
    metrics = {"Pass@1": np.mean(pass_at_1)}
    return metrics

# Evaluate zero-shot performance for baseline model
zero_shot_metrics_baseline = evaluate_zero_shot_metrics(questions, ground_truths, baseline_pipeline)
print("Zero-Shot Metrics (Baseline):", zero_shot_metrics_baseline)

# Evaluate zero-shot performance for fine-tuned model
zero_shot_metrics_finetuned = evaluate_zero_shot_metrics(questions, ground_truths, finetuned_pipeline)
print("Zero-Shot Metrics (Fine-Tuned):", zero_shot_metrics_finetuned)
