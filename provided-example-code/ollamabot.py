import ollama

MODAL = "gemma3:1b"

def chatbot():
    print(f"Welcome to the {MODAL} Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = ollama.generate(model=MODAL, prompt=user_input)
            print("Bot:", response['response'])
        except Exception as e:
            print("Error:", str(e))

if __name__ == "__main__":
    chatbot()