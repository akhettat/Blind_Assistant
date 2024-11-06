from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Function to download the BLOOM model
def download_bloom_model(model_name="bigscience/bloom-560m", save_directory="./bloom_model"):
    print("Downloading the BLOOM model. This may take some time depending on your internet connection.")
    # Load the tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save the model and tokenizer to the specified directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer have been saved to {save_directory}")

if __name__ == "__main__":
    download_bloom_model()

