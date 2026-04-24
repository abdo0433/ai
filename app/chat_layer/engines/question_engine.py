from app.chat_layer.core.llm_client import call_llm

def generate_question(job_title, job_description, cv_text, stage):
    prompt = f"""
Interview Stage: {stage}

Job Title:
{job_title}

Job Description:
{job_description}

Candidate CV:
{cv_text}

Task:
Generate ONE {stage} interview question suitable for a fresh graduate.
Return ONLY the question text, without any extra explanations.
"""
    response_text = call_llm(prompt)

    # Debug
    print("LLM raw question:", response_text)

    # Clean output to extract just the question text
    question = response_text.strip().split("\n")[0]
    return question
