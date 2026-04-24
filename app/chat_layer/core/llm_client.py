from groq import Groq
from app.chat_layer.config import GROQ_API_KEY, MODEL_NAME


SYSTEM_PROMPT = """
You are a professional technical interviewer for graduate-level candidates.
You conduct structured, fair, and realistic job interviews.

Rules:
- Ask only ONE question at a time.
- Be concise and professional.
- Match the question difficulty to a fresh graduate level.
- Do NOT provide hints or answers.
- Do NOT mention that you are an AI.
- Follow the interview stage strictly.

Your goal is to simulate a real interview experience.
"""

client = Groq(api_key=GROQ_API_KEY)
def call_llm(prompt: str) -> str:
    """
    Sends a prompt to the Groq LLM and returns the response content.
    Uses the SYSTEM_PROMPT for consistent behavior.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content
