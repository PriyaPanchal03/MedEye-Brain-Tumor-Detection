from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)
models = [
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash-lite",
    "models/gemini-flash-latest",
    "models/gemini-2.0-flash",
    "models/gemma-3-4b-it"
]

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
    return text.lower().strip() in greetings

def medical_chatbot(user_question):
    prompt = f"""
You are a medical AI assistant for a brain MRI system.

Rules:
- Answer in ONLY 3–4 short lines
- Use simple language
- Encourage consulting a medical professional
- Do NOT diagnose
- Do NOT suggest treatment

User Question:
{user_question}
"""

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            continue  # try next model

    return "⚠️ AI is temporarily unavailable. Please try again later."

