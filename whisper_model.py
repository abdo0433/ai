import whisper

# هيتحمل للمرة الأولى ويحفظ في app/models
model = whisper.load_model("base", download_root="app/models")
print("Whisper model loaded locally!")
