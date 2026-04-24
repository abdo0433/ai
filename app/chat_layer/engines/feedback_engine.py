from app.chat_layer.core.llm_client import call_llm
import json

def generate_feedback(evaluations):
    prompt = f"""
Interview Evaluations:
{evaluations}

IMPORTANT:
- Return ONLY valid JSON object.
- Do NOT add any extra text or explanations.
- JSON must include:
  overall_score (0-100),
  strengths (list),
  weaknesses (list),
  improvement_suggestions (list)
"""
    response_text = call_llm(prompt)

    # Debug
    print("LLM raw feedback:", response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON from LLM",
            "raw_response": response_text
        }
