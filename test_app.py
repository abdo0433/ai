"""
AI Interview Platform - Integration Test Suite
===============================================
Tests:
1. Server health check
2. Session start
3. Full interview simulation (all stages, adaptive questions)
4. Video frame submission
5. Final feedback
6. Quick analysis

Run: python test_interview.py
     python test_interview.py --url http://localhost:8000
     python test_interview.py --quick   # only health + quick test
"""

import requests
import json
import time
import sys
import argparse
import base64
import numpy as np
from typing import Optional, Dict

# ============================================================
# CONFIG
# ============================================================

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 45

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ============================================================
# TEST DATA
# ============================================================

JOB_TITLE = "Junior Python Developer"
JOB_DESCRIPTION = """
We are looking for a Junior Python Developer to join our team.
Requirements:
- Python (OOP, data structures, algorithms)
- Familiarity with REST APIs and web frameworks (Flask/FastAPI/Django)
- Version control with Git
- Problem-solving mindset
- Good communication skills
"""

CV_TEXT = """
Ahmed Mohamed
Computer Science Graduate, Cairo University 2024

Skills: Python, JavaScript, SQL, Git, FastAPI, Machine Learning basics
Projects:
- Built a REST API for a task management app using FastAPI
- Developed a sentiment analysis model using scikit-learn
- Contributed to open source Python libraries

Experience:
- Internship at Techno Corp (3 months) - Backend development with Python/Django

Languages: Arabic (native), English (professional)
"""

SAMPLE_ANSWERS = {
    "introduction": [
        "I'm Ahmed, a fresh Computer Science graduate from Cairo University. I'm passionate about backend development and have been working with Python for the past 3 years. I built several projects including a FastAPI-based REST service and a machine learning sentiment analysis tool. I applied because your company is known for innovation and I want to grow in a real production environment.",
        "I'm motivated by the opportunity to work with a professional team and apply what I've learned. I love problem-solving and Python makes it elegant. This role aligns perfectly with my skills and career goals.",
        "In 5 years I see myself as a mid-level backend engineer, possibly leading small projects. I want to deepen my knowledge in distributed systems and cloud architecture while contributing meaningfully to the team.",
    ],
    "technical": [
        "I once had a memory leak in a Django application. I used Python's memory_profiler to identify objects that weren't being garbage collected. Turned out we had circular references in our ORM models. I fixed it by restructuring the relationships and adding explicit cleanup in the relevant lifecycle methods.",
        "I'm most comfortable with Python and FastAPI. I chose FastAPI because it gives automatic OpenAPI docs, async support, and is very performant. I also use PostgreSQL for databases and Redis for caching.",
        "I start by reproducing the bug reliably. Then I use logging and debuggers like pdb or IDE debuggers to trace the execution flow. I isolate the problem to the smallest unit and write a test case that fails, then fix until the test passes.",
        "Imagine your computer is like a kitchen. An API is like a waiter — you tell the waiter what you want, the kitchen prepares it, and the waiter brings it back. The kitchen doesn't talk to you directly, only through the waiter. That's how APIs let different software systems communicate.",
    ],
    "behavioral": [
        "During my internship, we had a critical bug 2 days before a product launch. I volunteered to stay late and work with the senior developer to trace and fix it. We stayed until 11pm and resolved it. The product launched successfully on time.",
        "In a team project, a colleague and I disagreed on the database design. We both presented our approaches to the team and decided to benchmark both for performance. My colleague's design was 30% faster for read queries, so we used it. I learned to separate ego from technical decisions.",
        "In a hackathon, our team leader dropped out suddenly. I stepped up, redistributed tasks based on everyone's strengths, set clear milestones, and kept communication open. We finished and placed second overall.",
    ],
}

# ============================================================
# HELPERS
# ============================================================

def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def print_section(text: str):
    print(f"\n{CYAN}--- {text} ---{RESET}")

def ok(text: str):
    print(f"  {GREEN}✅ {text}{RESET}")

def fail(text: str):
    print(f"  {RED}❌ {text}{RESET}")

def warn(text: str):
    print(f"  {YELLOW}⚠️  {text}{RESET}")

def info(text: str):
    print(f"  {BLUE}ℹ️  {text}{RESET}")

def check(condition: bool, success_msg: str, fail_msg: str) -> bool:
    if condition:
        ok(success_msg)
        return True
    else:
        fail(fail_msg)
        return False


def make_fake_audio_features() -> Dict:
    """Generate realistic fake audio features."""
    return {
        "pitch_mean": float(np.random.uniform(120, 220)),
        "pitch_std": float(np.random.uniform(20, 50)),
        "pitch_variability": float(np.random.uniform(0.05, 0.2)),
        "energy_mean": float(np.random.uniform(0.1, 0.3)),
        "energy_std": float(np.random.uniform(0.02, 0.08)),
        "speaking_rate_estimate": float(np.random.uniform(110, 170)),
        "silence_ratio": float(np.random.uniform(0.1, 0.3)),
        "num_pauses": int(np.random.randint(3, 10)),
        "avg_pause_duration": float(np.random.uniform(0.3, 0.8)),
        "overall_audio_quality": float(np.random.uniform(65, 90)),
        "mfcc_mean": [float(x) for x in np.random.randn(13)],
        "mfcc_std": [float(x) for x in np.abs(np.random.randn(13))],
    }


def make_fake_video_features() -> Dict:
    """Generate realistic fake video features."""
    return {
        "emotions": {
            "happy": float(np.random.uniform(15, 35)),
            "neutral": float(np.random.uniform(40, 60)),
            "fear": float(np.random.uniform(3, 10)),
            "sad": float(np.random.uniform(2, 8)),
            "angry": float(np.random.uniform(1, 5)),
            "disgust": float(np.random.uniform(0, 3)),
            "surprised": float(np.random.uniform(3, 10)),
            "dominant_emotion": "neutral",
            "positive_ratio": float(np.random.uniform(0.25, 0.45)),
            "negative_ratio": float(np.random.uniform(0.05, 0.2)),
        },
        "gaze": {
            "average_gaze_x": 0.5,
            "average_gaze_y": 0.5,
            "gaze_stability": float(np.random.uniform(0.7, 0.95)),
            "eye_contact_percentage": float(np.random.uniform(60, 85)),
            "looking_away_percentage": float(np.random.uniform(10, 30)),
            "has_iris_data": True,
            "quality": "high",
        },
        "behavioral": {
            "confidence_score": float(np.random.uniform(60, 85)),
            "stress_level": float(np.random.uniform(20, 45)),
            "smile_percentage": float(np.random.uniform(15, 35)),
            "head_stability": float(np.random.uniform(0.7, 0.95)),
            "overall_composure": float(np.random.uniform(60, 85)),
        },
        "posture": {
            "average_posture_score": float(np.random.uniform(6, 9)),
            "slouching_percentage": float(np.random.uniform(5, 20)),
            "upright_percentage": float(np.random.uniform(70, 90)),
            "posture_notes": "Mostly upright posture",
        },
        "eye_contact": {
            "eye_contact_percentage": float(np.random.uniform(60, 85)),
            "eye_contact_quality": "good",
            "engagement_level": "high",
        },
        "gestures": {
            "fidgeting_detected": bool(np.random.random() > 0.7),
            "fidgeting_severity": "low",
            "gesture_notes": "Natural gestures",
        },
        "alerts": [],
        "frames_analyzed": 90,
        "duration": 30.0,
    }


def make_fake_frame_base64() -> str:
    """Generate a fake 64x64 BGR frame as base64 JPEG."""
    try:
        import cv2
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except ImportError:
        # cv2 not available, return empty pixel
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        return base64.b64encode(tiny.tobytes()).decode('utf-8')

# ============================================================
# TESTS
# ============================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add(self, passed: bool):
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print_header("TEST SUMMARY")
        print(f"  Total:  {total}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        if self.failed == 0:
            print(f"\n  {GREEN}{BOLD}🎉 All tests passed!{RESET}")
        else:
            print(f"\n  {YELLOW}⚠️  Some tests failed. Check logs above.{RESET}")
        return self.failed == 0


results = TestResults()


def test_health(base_url: str) -> bool:
    print_section("1. Health Check")
    try:
        r = requests.get(f"{base_url}/health", timeout=TIMEOUT)
        ok(f"Server responded: HTTP {r.status_code}")
        data = r.json()

        passed = check(r.status_code == 200, "Status 200 OK", f"Expected 200, got {r.status_code}")
        results.add(passed)

        passed = check(data.get("status") == "healthy", "Status is healthy", "Status not healthy")
        results.add(passed)

        engines = data.get("engines", {})
        for eng in ["llm", "audio", "video", "evaluation", "questions"]:
            val = engines.get(eng, False)
            if val:
                ok(f"Engine '{eng}' loaded")
            else:
                warn(f"Engine '{eng}' not loaded (will use fallback)")

        return True
    except requests.exceptions.ConnectionError:
        fail(f"Cannot connect to server at {base_url}")
        fail("Make sure the server is running: uvicorn main:app --reload --port 8000")
        results.add(False)
        return False
    except Exception as e:
        fail(f"Health check error: {e}")
        results.add(False)
        return False


def test_root(base_url: str):
    print_section("2. Root Endpoint")
    try:
        r = requests.get(f"{base_url}/", timeout=TIMEOUT)
        data = r.json()
        passed = check(r.status_code == 200, "Root endpoint OK", f"Got {r.status_code}")
        results.add(passed)
        info(f"API version: {data.get('version', 'unknown')}")
        info(f"Active sessions: {data.get('active_sessions', 0)}")
    except Exception as e:
        fail(f"Root endpoint error: {e}")
        results.add(False)


def test_engine_status(base_url: str):
    print_section("3. Engine Status")
    try:
        r = requests.get(f"{base_url}/api/engines/status", timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Engine status OK", f"Got {r.status_code}")
        results.add(passed)
        data = r.json()
        for eng, details in data.items():
            loaded = details.get("loaded", False)
            if loaded:
                ok(f"{eng}: loaded")
            else:
                warn(f"{eng}: not loaded")
    except Exception as e:
        fail(f"Engine status error: {e}")
        results.add(False)


def test_quick_analysis(base_url: str):
    print_section("4. Quick Analysis (no session)")
    try:
        payload = {
            "question": "Tell me about yourself.",
            "answer": "I am a software engineer with 2 years experience in Python and FastAPI. I enjoy building scalable backend systems.",
            "audio_features": make_fake_audio_features(),
            "video_features": make_fake_video_features(),
        }
        r = requests.post(f"{base_url}/api/analyze/quick", json=payload, timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Quick analysis OK", f"Got {r.status_code}")
        results.add(passed)

        data = r.json()
        passed = check(data.get("success", False), "Response success=True", "success not True")
        results.add(passed)

        eval_data = data.get("evaluation", {})
        for field in ["technical_score", "communication_score", "short_feedback"]:
            if field in eval_data:
                ok(f"Field '{field}' present: {str(eval_data[field])[:60]}")
            else:
                warn(f"Field '{field}' missing from evaluation")

    except Exception as e:
        fail(f"Quick analysis error: {e}")
        results.add(False)


def test_full_interview(base_url: str):
    """Simulate a complete interview session."""
    print_section("5. Full Interview Session")

    # --- Start session ---
    print(f"  {YELLOW}Starting session...{RESET}")
    try:
        payload = {
            "job_title": JOB_TITLE,
            "job_description": JOB_DESCRIPTION,
            "cv_text": CV_TEXT,
        }
        r = requests.post(f"{base_url}/api/session/start", json=payload, timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Session started", f"Got {r.status_code}")
        results.add(passed)
        if not passed:
            fail("Cannot continue without a session")
            return None

        data = r.json()
        session_id = data.get("session_id")
        passed = check(bool(session_id), "Session ID received", "No session_id")
        results.add(passed)

        first_question = data.get("first_question", "")
        passed = check(bool(first_question), f"First question received", "No question")
        results.add(passed)

        info(f"Session ID: {session_id}")
        info(f"Stage: {data.get('stage')} ({data.get('question_number')}/{data.get('total_questions')})")
        info(f"Q: {first_question[:100]}...")

    except Exception as e:
        fail(f"Session start error: {e}")
        results.add(False)
        return None

    # --- Submit video frame test ---
    print(f"\n  {YELLOW}Testing video frame submission...{RESET}")
    try:
        frame_payload = {
            "session_id": session_id,
            "frame_base64": make_fake_frame_base64(),
        }
        r = requests.post(
            f"{base_url}/api/session/{session_id}/video-frame",
            json=frame_payload,
            timeout=TIMEOUT,
        )
        passed = check(r.status_code == 200, "Video frame submitted", f"Got {r.status_code}")
        results.add(passed)
        if passed:
            info(f"Buffered frames: {r.json().get('buffered_frames', 0)}")
    except Exception as e:
        warn(f"Video frame test error: {e}")

    # --- Get session status ---
    print(f"\n  {YELLOW}Checking session status...{RESET}")
    try:
        r = requests.get(f"{base_url}/api/session/{session_id}/status", timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Session status OK", f"Got {r.status_code}")
        results.add(passed)
        status_data = r.json()
        info(f"Status: {status_data.get('status')}")
        info(f"Current stage: {status_data.get('current_stage')}")
    except Exception as e:
        warn(f"Status check error: {e}")

    # --- Run through all interview stages ---
    print(f"\n  {YELLOW}Running full interview simulation...{RESET}")

    stage_index = 0
    total_answered = 0
    current_question = first_question

    all_stages = ["introduction", "technical", "behavioral"]
    answer_pool = SAMPLE_ANSWERS.copy()
    used_answers = {s: 0 for s in all_stages}

    interview_complete = False

    while not interview_complete:
        current_stage = all_stages[min(stage_index, len(all_stages) - 1)]

        # Pick an answer
        answers_for_stage = answer_pool.get(current_stage, ["I think this is a good question and I would approach it systematically."])
        answer_idx = used_answers[current_stage] % len(answers_for_stage)
        answer = answers_for_stage[answer_idx]
        used_answers[current_stage] += 1

        print(f"\n  {BOLD}[{current_stage.upper()}] Q: {current_question[:80]}...{RESET}")
        print(f"  A: {answer[:80]}...")

        try:
            submit_payload = {
                "session_id": session_id,
                "answer": answer,
                "audio_features": make_fake_audio_features(),
                "video_features": make_fake_video_features(),
            }
            r = requests.post(
                f"{base_url}/api/session/{session_id}/answer",
                json=submit_payload,
                timeout=TIMEOUT,
            )

            passed = check(r.status_code == 200, f"Answer {total_answered+1} submitted", f"Got {r.status_code}")
            results.add(passed)

            if not passed:
                fail("Stopping interview simulation due to error")
                break

            resp = r.json()
            eval_data = resp.get("evaluation", {})
            tech = eval_data.get("technical_score", "?")
            comm = eval_data.get("communication_score", "?")
            ok(f"  Scores — Tech: {tech}, Comm: {comm}")

            total_answered += 1
            interview_complete = resp.get("interview_complete", False)
            stage_complete = resp.get("stage_complete", False)

            if interview_complete:
                ok("Interview complete!")
                break

            if stage_complete:
                stage_index += 1
                ok(f"Stage complete! Moving to: {resp.get('stage')}")

            current_question = resp.get("next_question", "")
            if not current_question:
                warn("No next question received")
                break

            # Small delay to avoid hammering
            time.sleep(3.0)

        except Exception as e:
            fail(f"Answer submission error: {e}")
            results.add(False)
            break

    info(f"Total questions answered: {total_answered}")

    # --- Final feedback ---
    print(f"\n  {YELLOW}Getting final feedback...{RESET}")
    try:
        r = requests.get(f"{base_url}/api/session/{session_id}/feedback", timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Final feedback received", f"Got {r.status_code}")
        results.add(passed)

        if passed:
            fb = r.json()
            ok(f"Overall score: {fb.get('overall_score')}/100")
            ok(f"Technical: {fb.get('technical_score')}/100")
            ok(f"Communication: {fb.get('communication_score')}/100")
            ok(f"Confidence: {fb.get('confidence_index')}/100")
            ok(f"Recommendation: {fb.get('recommendation')}")

            strengths = fb.get("strengths", [])
            weaknesses = fb.get("weaknesses", [])
            suggestions = fb.get("improvement_suggestions", [])

            info(f"Strengths ({len(strengths)}): {', '.join(strengths[:3])}")
            info(f"Weaknesses ({len(weaknesses)}): {', '.join(weaknesses[:3])}")
            info(f"Suggestions ({len(suggestions)}): {', '.join(suggestions[:2])}")

            stage_sums = fb.get("stage_summaries", {})
            for stage, summary in stage_sums.items():
                info(f"  {stage}: tech={summary.get('avg_technical')}, comm={summary.get('avg_communication')}, q={summary.get('questions_answered')}")

    except Exception as e:
        fail(f"Final feedback error: {e}")
        results.add(False)

    # --- Cleanup ---
    print(f"\n  {YELLOW}Cleaning up session...{RESET}")
    try:
        r = requests.delete(f"{base_url}/api/session/{session_id}", timeout=TIMEOUT)
        passed = check(r.status_code == 200, "Session deleted", f"Got {r.status_code}")
        results.add(passed)
    except Exception as e:
        warn(f"Cleanup error: {e}")

    return session_id


def test_error_cases(base_url: str):
    print_section("6. Error Cases")

    # Non-existent session
    try:
        r = requests.get(f"{base_url}/api/session/fake-session-id/status", timeout=TIMEOUT)
        passed = check(r.status_code == 404, "404 for missing session", f"Expected 404, got {r.status_code}")
        results.add(passed)
    except Exception as e:
        fail(f"Error case test failed: {e}")
        results.add(False)

    # Missing job title
    try:
        r = requests.post(
            f"{base_url}/api/session/start",
            json={"job_title": "", "job_description": "test", "cv_text": ""},
            timeout=TIMEOUT,
        )
        passed = check(r.status_code in [400, 422], "Validation error for empty job_title",
                       f"Expected 400/422, got {r.status_code}")
        results.add(passed)
    except Exception as e:
        fail(f"Validation test failed: {e}")
        results.add(False)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AI Interview Platform Test Suite")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL (default: {DEFAULT_URL})")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests (health + quick analysis)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print_header(f"AI INTERVIEW PLATFORM - TEST SUITE")
    print(f"  Server: {base_url}")
    print(f"  Mode:   {'Quick' if args.quick else 'Full'}")

    # --- Health check first ---
    server_ok = test_health(base_url)
    if not server_ok:
        print(f"\n{RED}Server not reachable. Stopping tests.{RESET}")
        sys.exit(1)

    test_root(base_url)
    test_engine_status(base_url)
    test_quick_analysis(base_url)

    if not args.quick:
        test_full_interview(base_url)
        test_error_cases(base_url)

    # Print summary
    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()