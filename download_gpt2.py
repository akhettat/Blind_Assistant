from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle GPT-2 depuis Hugging Face
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Générer du texte
prompt = "Bonjour, assistant."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
