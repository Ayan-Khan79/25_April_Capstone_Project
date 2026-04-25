import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_AI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_AI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")

def get_ai_response(user_query: str):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: Could not retrieve AI response. {str(e)}"