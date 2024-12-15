# Step 2: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


# Step 2: Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Step 3: Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can also use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# Step 4 Ensure that the tokenizer knows the padding token
tokenizer.pad_token = tokenizer.eos_token


# Move model to GPU if available
model = model.to(device)


# Step 5: Define function for chat completion
def chat_with_gpt2(input_text, max_length=100, temperature=0.7, top_p=0.9):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
   
    # Generate the model output
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Prevents repetition of the same n-grams
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
   
    # Decode the generated tokens and return as text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Step 6: Test chat completion
input_text = "Hello, how are you today?"
response = chat_with_gpt2(input_text)
print("GPT-2 Response:", response)


# Step 7: Optional chat loop (for interactive chat)
print("Chat with GPT-2! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Exiting the chat.")
        break
   
    # Get the response from GPT-2
    response = chat_with_gpt2(user_input)
   
    # Print the model's response
    print(f"GPT-2: {response}")
