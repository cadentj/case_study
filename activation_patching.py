from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
import os

def full_rome_causal_tracing(model, prompt):
    prompt = model.tokenizer.encode(prompt)
    answer_token = model.tokenizer.encode(" Seattle")
    layers = model.transformer.h
    lm_head = model.lm_head
    embedding = model.transformer.wte
    with model.trace() as tracer:
        clean_acts = []
        with tracer.invoke(prompt):
            for layer in layers:
                clean_acts.append(layer.output[0])
            # Save probability over the answer token
            probs = lm_head.output.softmax(-1)
            clean_value = probs[:,-1, answer_token]
        results = []
        for token in range(len(prompt)):
            per_token_result = []
            for layer_idx, layer in enumerate(layers):
                with tracer.invoke(prompt):
                    # Corrupt the subject tokens
                    embedding.output[:,0:3,:][:] = 0.
                    # Restore the clean activations
                    clean_token_act = clean_acts[layer_idx][:,token,:]
                    layer.output[0][:,token,:] = clean_token_act
                    # Save probability over the answer token
                    probs = lm_head.output.softmax(-1)
                    # Compute difference in clean and corrupted
                    corrupted_value = probs[:,-1, answer_token]
                    diff = clean_value - corrupted_value
                    per_token_result.append(diff.item().save())
            results.append(per_token_result)

# Split into separate traces as not to overload CUDA memory
def low_memory_rome_causal_tracing(model, prompt, target):

    clean_acts = []
    n_layers = len(model.transformer.h)
    n_tokens = len(prompt)
    
    with model.trace(prompt):
        for i, layer in enumerate(model.transformer.h):
            clean_acts.append(layer.output[0].save())

        probs = model.lm_head.output.softmax(-1)
        clean_value = probs[:,-1, target]
        clean_value.save()

    results = []
    clean_value = clean_value.value.item()

    for t in range(n_tokens):

        per_token_result = []
        for layer in range(n_layers):
            with model.trace(prompt, scan=False):
                # Corrupt the subject tokens
                model.transformer.wte.output[:,0:3,:][:] = 0.

                # Restore the clean activations
                model.transformer.h[layer].output[0][:,t,:] = clean_acts[layer][:,t,:]
                
                probs = model.lm_head.output.softmax(-1)
                difference = clean_value - probs[:,-1, target]
                
                per_token_result.append(difference.item().save())                   

        results.append(per_token_result)

    return results

def plot(results, y_labels, save_name="plot.png"):

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(results, cmap='Purples_r', aspect='auto')
    fig.colorbar(cax, ax=ax, orientation='vertical')

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('single restored layer within GPT-2-XL')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, save_name)
    plt.savefig(save_path)

def cast_value_array_to_real(results):
    # Call .value to not throw matplot errors
    new_results = []
    for i in results:
        row = []
        for j in i:
            temp = j.value
            row.append(temp)
        new_results.append(row)

    return new_results

############### BODY ###############

# Declare model and tokenizer
model = LanguageModel("openai-community/gpt2-xl", device_map="auto")
tokenizer = model.tokenizer

# Define prompt 
prompt = "The Space Needle is in downtown"
tokenized_prompt = tokenizer.encode(prompt)
str_tokens = [tokenizer.decode(t) for t in tokenized_prompt]

# Define target token
target = " Seattle"
target_token = tokenizer.encode(target)

# Run experiments
results = low_memory_rome_causal_tracing(model, tokenized_prompt, target_token)
results = cast_value_array_to_real(results)
plot(results, str_tokens)

