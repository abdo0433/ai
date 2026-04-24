"""
Realtime Interview Orchestrator
================================
Responsibilities:
- Connect Audio subsystem with Interview Flow Controller
- Manage interview session in voice mode
- Coordinate between STT, TTS, and Flow Controller

Does NOT:
- Contain interview logic
- Evaluate answers
- Generate questions
"""

from typing import Optional, Dict
from uuid import uuid4

from app.chat_layer.models.interview import InterviewSession, InterviewState
from app.chat_layer.engines.stt_engine import get_stt_engine
from app.chat_layer.engines.tts_engine import get_tts_engine
from app.chat_layer.engines.question_engine import generate_question
from app.chat_layer.engines.evaluation_engine_multimodel import evaluate_answer
from app.chat_layer.engines.feedback_engine import generate_feedback
from app.chat_layer.core.flow_controller import next_state


class RealtimeInterviewOrchestrator:
    """
    Orchestrates voice-based interview sessions.
    Connects audio I/O with interview logic.
    """
    
    def __init__(
        self,
        session: InterviewSession,
        whisper_model: str = "base",
        tts_rate: int = 150,
        tts_volume: float = 0.9
    ):
        """
        Initialize orchestrator for a specific interview session.
        
        Args:
            session: InterviewSession instance
            whisper_model: Whisper model size
            tts_rate: TTS speech rate
            tts_volume: TTS volume
        """
        self.session = session
        
        # Initialize audio engines
        self.stt = get_stt_engine(model_path=whisper_model, language="en")
        self.tts = get_tts_engine(rate=tts_rate, volume=tts_volume)
        
        print(f"✓ Realtime Interview Orchestrator initialized for session: {session.session_id}")
    
    def start_interview(self) -> None:
        """
        Start the voice interview process.
        """
        print("\n" + "="*60)
        print("🎤 VOICE INTERVIEW STARTED")
        print("="*60)
        
        # Generate first question
        first_question = generate_question(
            self.session.job_title,
            self.session.job_description,
            self.session.cv_text,
            self.session.state.value
        )
        
        self.session.questions.append(first_question)
        
        # Speak the first question
        print(f"\n📝 Question: {first_question}")
        self.tts.speak(first_question, blocking=True)  # Wait for speech to finish
        
        import time
        time.sleep(1)  # Give user a moment to prepare
        
        # Main interview loop
        self._interview_loop()
    
    def _interview_loop(self) -> None:
        """
        Main interview loop: Listen -> Transcribe -> Evaluate -> Ask Next
        """
        import time
        
        while self.session.state != InterviewState.ENDED:
            # Get current question
            current_question = self.session.questions[-1]
            
            # Listen to user's answer
            print("\n🎤 Your turn to speak... (You have 1 minute to answer)")
            print("💡 Speak clearly and pause for 3-4 seconds when you're done.")
            user_answer = self.stt.listen_and_transcribe(max_duration=60)
            
            # Handle empty/failed transcription
            if not user_answer or len(user_answer.strip()) < 3:
                retry_message = "I didn't catch that clearly. Could you please repeat your answer?"
                print(f"⚠️  {retry_message}")
                self.tts.speak(retry_message, blocking=True)
                time.sleep(1)
                continue
            
            # Store answer
            self.session.answers.append(user_answer)
            print(f"✓ Your answer: {user_answer}")
            
            # Evaluate answer
            print("🔄 Evaluating answer...")
            evaluation = evaluate_answer(current_question, user_answer)
            self.session.evaluations.append(evaluation)
            
            # Optional: Speak brief acknowledgment
            acknowledgment = self._generate_acknowledgment(evaluation)
            if acknowledgment:
                print(f"💬 {acknowledgment}")
                self.tts.speak(acknowledgment, blocking=True)
                time.sleep(0.5)
            
            # Move to next state
            self.session.state = next_state(self.session.state)
            
            # Check if interview ended
            if self.session.state == InterviewState.ENDED:
                self._end_interview()
                break
            
            # Generate next question
            next_question = generate_question(
                self.session.job_title,
                self.session.job_description,
                self.session.cv_text,
                self.session.state.value
            )
            
            self.session.questions.append(next_question)
            
            # Speak next question with blocking
            print(f"\n📝 Next Question: {next_question}")
            self.tts.speak(next_question, blocking=True)
            time.sleep(1)  # Give user time to prepare
    
    def _generate_acknowledgment(self, evaluation: Dict) -> Optional[str]:
        """
        Generate brief acknowledgment based on evaluation.
        
        Args:
            evaluation: Evaluation dict from evaluation_engine
            
        Returns:
            Acknowledgment text or None
        """
        # Check if evaluation has error
        if "error" in evaluation:
            return None
        
        # Simple acknowledgments based on score
        try:
            avg_score = (
                evaluation.get("technical_score", 0) +
                evaluation.get("communication_score", 0) +
                evaluation.get("relevance_score", 0)
            ) / 3
            
            if avg_score >= 75:
                return "Good answer. Let's move on."
            elif avg_score >= 50:
                return "Thank you. Next question."
            else:
                return "I see. Let's continue."
                
        except Exception:
            return "Thank you."
    
    def _end_interview(self) -> None:
        """
        End the interview and provide feedback.
        """
        print("\n" + "="*60)
        print("🏁 INTERVIEW COMPLETED")
        print("="*60)
        
        # Generate final feedback
        print("🔄 Generating feedback...")
        feedback = generate_feedback(self.session.evaluations)
        
        # Speak ending message
        ending_message = "Thank you for your time. The interview is now complete. You will receive detailed feedback shortly."
        print(f"\n💬 {ending_message}")
        self.tts.speak(ending_message, blocking=True)
        
        # Print feedback (don't speak it all - too long)
        print("\n📊 FEEDBACK:")
        print(feedback)
        
        # Save feedback to session
        self.session.feedback = feedback


# Factory function for easy creation
def create_realtime_interview(
    job_title: str,
    job_description: str,
    cv_text: str,
    whisper_model: str = "base"
) -> RealtimeInterviewOrchestrator:
    """
    Factory function to create a new realtime interview session.
    
    Args:
        job_title: Job title
        job_description: Job description
        cv_text: Candidate's CV text
        whisper_model: Whisper model size
        
    Returns:
        RealtimeInterviewOrchestrator instance
    """
    # Create session
    session_id = str(uuid4())
    session = InterviewSession(
        session_id=session_id,
        job_title=job_title,
        job_description=job_description,
        cv_text=cv_text
    )
    
    # Create orchestrator
    orchestrator = RealtimeInterviewOrchestrator(
        session=session,
        whisper_model=whisper_model
    )
    
    return orchestrator


# Simple CLI interface for testing
def run_voice_interview_cli():
    """
    Simple CLI interface for testing voice interviews.
    """
    print("="*60)
    print("🎙️  AI VOICE INTERVIEW SYSTEM")
    print("="*60)
    
    # Get job details
    job_title = input("\nJob Title: ").strip()
    job_description = input("Job Description: ").strip()
    cv_text = input("Your CV Summary: ").strip()
    
    # Create and start interview
    orchestrator = create_realtime_interview(
        job_title=job_title,
        job_description=job_description,
        cv_text=cv_text
    )
    
    orchestrator.start_interview()
    
    print("\n" + "="*60)
    print("✓ Interview session completed!")
    print("="*60)


if __name__ == "__main__":
    run_voice_interview_cli()