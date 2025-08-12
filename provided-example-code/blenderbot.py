from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill" # 750MB
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Function to generate a response
def get_response(user_input):
    # Encode the input
    inputs = tokenizer([user_input], return_tensors="pt")

    # Generate response
    reply_ids = model.generate(**inputs)

    # Decode and return the response
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

# Simple chat loop
print("BlenderBot: Hi! How can I help you today? (Type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = get_response(user_input)
    print(f"BlenderBot: {response}")