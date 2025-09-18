from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def getClient() -> OpenAI:
    return OpenAI()

def getUserInput() -> str:
    return input("Type in your prompt: ")

def askModel(client: OpenAI, model: str, message: str) -> str:
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message}
        ]
    )

def main():
    while True:
        user_input = getUserInput()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the program.")
            break

        client = getClient()
        response = askModel(client, "gpt-3.5-turbo", user_input)
        print("Response from model:", response.choices[0].message.content)
        
        
        
if __name__ == "__main__":
    main()