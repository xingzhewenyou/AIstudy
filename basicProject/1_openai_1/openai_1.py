import openai

# Set OpenAI API key
openai.api_key = api_key

def verify_api_key():
    try:
        # Make a simple API call to verify the key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, OpenAI!"}]
        )
        print("API key is valid.")
    except openai.error.AuthenticationError:
        print("Invalid API key.")
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please check your plan and billing details.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    verify_api_key()