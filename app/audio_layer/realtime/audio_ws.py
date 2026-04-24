from fastapi import APIRouter, WebSocket
import pyttsx3

router = APIRouter()

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

@router.websocket("/ws/audio/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"WebSocket connected! Session: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            print("Received from client:", data)

            # TTS
            response_text = f"Server says: {data}"
            engine.say(response_text)
            engine.runAndWait()

            await websocket.send_text(response_text)

    except Exception as e:
        print("WebSocket closed:", e)
