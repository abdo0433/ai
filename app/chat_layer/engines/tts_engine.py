"""
Text-to-Speech Engine (Web Ready)
=================================
تحويل النص إلى صوت وإرجاعه كـ Base64 ليعمل في المتصفح
"""

from gtts import gTTS
import io
import base64
from typing import Optional

class TTSEngine:
    """
    محرك تحويل النص إلى كلام متوافق مع الويب.
    """
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        print(f"✓ TTS Engine initialized (lang={lang})")

    def generate_base64(self, text: str) -> str:
        """
        تحويل النص إلى ملف صوتي وإرجاعه كـ Base64 string للفرونت إند.
        """
        if not text or len(text.strip()) == 0:
            print("⚠️ Empty text, skipping TTS")
            return ""
        
        try:
            # استخدام gTTS لتوليد الصوت في الذاكرة
            tts = gTTS(text=text, lang=self.lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # تحويل البايتات لـ Base64 لتعمل كـ Data URI في المتصفح
            audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
            return f"data:audio/mpeg;base64,{audio_base64}"
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return ""

# Singleton instance
_tts_instance: Optional[TTSEngine] = None

def get_tts_engine(lang: str = 'en') -> TTSEngine:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine(lang=lang)
    return _tts_instance

# دالة الاستدعاء البسيطة التي ستستخدمها في interview_routes.py
def generate_tts_base64(text: str, lang: str = 'en') -> str:
    engine = get_tts_engine(lang=lang)
    return engine.generate_base64(text)