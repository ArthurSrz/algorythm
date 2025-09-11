import openai
import torch
from transformers import pipeline

# ============================================================================ #
# OpenAI configuration
openai.api_key = "sk-proj-6kqmfyh64EAJpvSJ8dGvT3BlbkFJiF5kN4QkzMcFLqe7IHip"

# ============================================================================ #
# Hugging Face TinyLlama configuration
try:
    print("[llm.py] Initializing Hugging Face pipeline with model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("[llm.py] Hugging Face pipeline initialized successfully!")
except Exception as e:
    print(f"[llm.py] Failed to initialize the Hugging Face pipeline: {e}")
    raise

# ============================================================================ #
# Function to generate response
def generate_response(question, context="", model="gpt-3.5-turbo"):
    try:
        if "gpt" in model:
            # OpenAI GPT model implementation
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question + (f". Use the following information: {context}" if len(context) > 0 else "")}
            ]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=3000,
                temperature=0  # Set temperature to 0 for deterministic output
            )
            return response.choices[0].message['content'].strip()
        elif "tinyllama" in model:
            # Hugging Face TinyLlama implementation
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": question + (f" Context: {context}" if context else "")},
            ]
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            # Extract only the <assistant> answer
            generated_text = outputs[0]["generated_text"].strip()
            if "<|assistant|>" in generated_text:
                return generated_text.split("<|assistant|>")[1].split("</s>")[0].strip()
            return "[llm.py] Failed to parse TinyLlama response."
        else:
            return "[llm.py] Unsupported model. Please use 'gpt-3.5-turbo' or 'tinyllama'."
    except Exception as e:
        print(f"[llm.py] An error occurred while generating the response: {e}")
        return "[llm.py] An error occurred while generating the response."
# ============================================================================ #
# Example usage
if __name__ == "__main__":
    # Example with OpenAI GPT
    question = "What is the capital of France?"
    response = generate_response(question, model="gpt-3.5-turbo")
    print(f"OpenAI GPT Response: {response}")

    # Example with TinyLlama
    question = "How many helicopters can a human eat in one sitting?"
    response = generate_response(question, style="pirate", model="tinyllama")
    print(f"TinyLlama Response: {response}")