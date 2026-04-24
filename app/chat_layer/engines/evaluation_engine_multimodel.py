"""
Enhanced Multi-modal Evaluation Engine
========================================
Evaluates interview answers using:
1. Text content  (what was said)
2. Audio features (how it was said — voice)
3. Video features (how it was said — body language, emotions, posture)

Fix applied:
  - json.loads replaced with the "slice" method so the engine
    handles responses wrapped in markdown fences (```json ... ```)
    without raising JSONDecodeError.
"""

from typing import Dict, Optional
import json


# ============================================================
# PUBLIC API
# ============================================================

def evaluate_answer(
    question:       str,
    answer:         str,
    audio_features: Optional[Dict] = None,
    video_features: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate an interview answer with optional audio and video features.

    Args:
        question:       The interview question.
        answer:         The candidate's transcribed answer.
        audio_features: Dict returned by AudioFeatureExtractor.extract_features()
        video_features: Dict returned by VideoFeatureExtractor.extract_features_from_video()

    Returns:
        Dict with scores, summary, strengths, weaknesses, and recommendations.
        On parse failure returns {"error": ..., "raw_response": ...}.
    """
    from app.chat_layer.core.llm_client import call_llm

    prompt        = _build_behavioral_evaluation_prompt(question, answer, audio_features, video_features)
    response_text = call_llm(prompt)

    print("LLM raw evaluation:", response_text)

    # ── Robust JSON extraction ──────────────────────────────
    # The LLM sometimes wraps the JSON in markdown fences.
    # Slicing from the first '{' to the last '}' handles that.
    start = response_text.find("{")
    end   = response_text.rfind("}") + 1

    if start == -1 or end == 0:
        return {"error": "No JSON object found in LLM response", "raw_response": response_text}

    try:
        evaluation = json.loads(response_text[start:end])
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}", "raw_response": response_text}

    # Attach human-readable summaries
    if audio_features:
        evaluation["audio_summary"] = _summarize_audio_features(audio_features)
    if video_features:
        evaluation["video_summary"] = _summarize_video_features(video_features)

    return evaluation


# ============================================================
# PROMPT BUILDER
# ============================================================

def _build_behavioral_evaluation_prompt(
    question:       str,
    answer:         str,
    audio_features: Optional[Dict],
    video_features: Optional[Dict],
) -> str:

    has_audio = audio_features is not None
    has_video = video_features is not None

    # ── System role ────────────────────────────────────────
    prompt  = "You are an advanced AI Behavioral Interview Evaluation System.\n"
    prompt += "You receive **multi-modal candidate data** including audio"
    if has_video:
        prompt += " and video features"
    prompt += ".\n\n"

    # ── Audio block ────────────────────────────────────────
    if has_audio:
        prompt += (
            f"Audio Features:\n"
            f"- Energy: {audio_features.get('energy_mean', 0):.3f}\n"
            f"- Pitch mean: {audio_features.get('pitch_mean', 0):.1f} Hz\n"
            f"- Pitch variability: {audio_features.get('pitch_variability', 0):.3f}\n"
            f"- Speaking rate: {audio_features.get('speaking_rate_estimate', 0):.0f} syllables/min\n"
            f"- Silence ratio: {audio_features.get('silence_ratio', 0):.2f}\n\n"
        )

    # ── Video block ────────────────────────────────────────
    if has_video:
        emotions   = video_features.get("emotions",   {})
        posture    = video_features.get("posture",     {})
        eye_contact= video_features.get("eye_contact", {})
        gestures   = video_features.get("gestures",    {})

        dom      = emotions.get("dominant_emotion", "neutral")
        happy    = emotions.get("happy",   0)
        neutral  = emotions.get("neutral", 0)
        fear     = emotions.get("fear",    0)
        emotion_str = f"{dom} ({happy:.0f}% happy, {neutral:.0f}% neutral, {fear:.0f}% anxious)"

        eye_pct     = eye_contact.get("eye_contact_percentage", 0)
        eye_quality = eye_contact.get("eye_contact_quality",    "fair")
        eye_str     = f"{eye_quality} ({eye_pct:.0f}% maintained)"

        prompt += (
            f"Video Features:\n"
            f"- Dominant emotions: {emotion_str} (detected using MTCNN, FER, DeepFace)\n"
            f"- Average posture score: {posture.get('average_posture_score', 0):.1f}/10 (MediaPipe)\n"
            f"- Eye contact and engagement: {eye_str}\n"
            f"- Gesture notes: {gestures.get('gesture_notes', 'Natural body language')}\n\n"
        )

    # ── Interview content ──────────────────────────────────
    prompt += f"Interview Content:\nQuestion: {question}\nAnswer: {answer}\n\n"

    # ── Tasks ──────────────────────────────────────────────
    modalities = "audio and video" if (has_audio and has_video) else ("video" if has_video else "audio")

    prompt += (
        f"Tasks:\n"
        f"1. Compute Unified Behavioral Scores using {modalities} metrics:\n"
        f"   - Confidence Index (0-100)\n"
        f"   - Stress Index (0-100)\n"
        f"   - Professional Presence Index (0-100)\n\n"
        f"2. Write a professional Summary paragraph (3-5 sentences) based on {modalities} features.\n\n"
        f"3. Identify Strengths with concrete behavioral indicators from {modalities} analysis.\n\n"
        f"4. Identify Weaknesses with actionable advice.\n\n"
        f"5. Evaluate content quality (technical accuracy, relevance, clarity).\n\n"
        f"6. Provide an Overall Recommendation: Strong Fit | Moderate Fit | Needs Improvement.\n\n"
        f"7. Return ONLY valid JSON — no markdown fences, no extra text — in this exact shape:\n"
        "{\n"
        '  "confidence_index": <0-100>,\n'
        '  "stress_index": <0-100>,\n'
        '  "professional_presence_index": <0-100>,\n'
        '  "summary": "<3-5 sentences>",\n'
        '  "strengths": ["<strength with indicator>", ...],\n'
        '  "weaknesses": ["<weakness with advice>", ...],\n'
        '  "technical_score": <0-100>,\n'
        '  "communication_score": <0-100>,\n'
        '  "relevance_score": <0-100>,\n'
        '  "short_feedback": "<brief content feedback>",\n'
        f'  "delivery_feedback": "<voice delivery{"  and body language" if has_video else ""} feedback>",\n'
        '  "overall_recommendation": "<Strong Fit | Moderate Fit | Needs Improvement>",\n'
        '  "recommendation_justification": "<1-2 sentences>"\n'
        "}\n\n"
    )

    # ── Final instructions ─────────────────────────────────
    if has_video:
        prompt += "- Use emotion %, posture, eye contact, and gestures to assess presence.\n"
    if has_audio:
        prompt += "- Use audio metrics to assess confidence, stress, and engagement.\n"
    prompt += (
        "- Focus on patterns across the whole response, not isolated moments.\n"
        "- Keep the report concise and professional.\n"
        "- Return ONLY valid JSON, nothing else.\n"
    )

    return prompt


# ============================================================
# SUMMARY HELPERS  (attached to evaluation dict)
# ============================================================

def _summarize_audio_features(audio: Dict) -> Dict:
    pv      = audio.get("pitch_variability",      0)
    energy  = audio.get("energy_mean",             0)
    rate    = audio.get("speaking_rate_estimate",  0)
    silence = audio.get("silence_ratio",           0)

    return {
        "tone":   "monotone"       if pv      < 0.05  else ("well-varied"   if pv      < 0.15  else "very expressive"),
        "volume": "quiet"          if energy  < 0.1   else ("good"          if energy  < 0.3   else "energetic"),
        "pace":   "slow"           if rate    < 90    else ("normal"        if rate    < 150   else ("fast" if rate < 200 else "very fast")),
        "pauses": "few"            if silence < 0.1   else ("natural"       if silence < 0.3   else "many"),
    }


def _summarize_video_features(video: Dict) -> Dict:
    emotions    = video.get("emotions",    {})
    posture     = video.get("posture",     {})
    eye_contact = video.get("eye_contact", {})
    gestures    = video.get("gestures",    {})

    return {
        "dominant_emotion":   emotions.get("dominant_emotion",     "neutral"),
        "posture_quality":    posture.get("posture_consistency",    "fair"),
        "eye_contact_quality":eye_contact.get("eye_contact_quality","fair"),
        "engagement_level":   eye_contact.get("engagement_level",   "moderate"),
        "fidgeting":          gestures.get("fidgeting_detected",    False),
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def evaluate_answer_audio_only(
    question:       str,
    answer:         str,
    audio_features: Optional[Dict] = None,
) -> Dict:
    return evaluate_answer(question, answer, audio_features=audio_features, video_features=None)