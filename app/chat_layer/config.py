import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 =os.getenv( "GROQ_API_KEY2") 
MODEL_NAME = "llama-3.1-8b-instant"
# MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL = "base"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 200
# Audio
SAMPLE_RATE = 16000
DURATION = 3 
WS_URI = "ws://127.0.0.1:8000/ws/audio/test"