"""
Interview Data Models
=====================
Core data structures for interview sessions.
"""

from enum import Enum
from typing import List, Optional, Dict


class InterviewState(Enum):
    """Interview progression states"""
    INTRO = "intro"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"
    ENDED = "ended"


class InterviewSession:
    """
    Represents a complete interview session.
    Stores all questions, answers, evaluations, and final feedback.
    """
    
    def __init__(
        self,
        session_id: str,
        job_title: str,
        job_description: str,
        cv_text: str
    ):
        """
        Initialize a new interview session.
        
        Args:
            session_id: Unique session identifier
            job_title: Target job title
            job_description: Job description/requirements
            cv_text: Candidate's CV text
        """
        self.session_id: str = session_id
        self.job_title: str = job_title
        self.job_description: str = job_description
        self.cv_text: str = cv_text
        
        # Interview progression
        self.state: InterviewState = InterviewState.INTRO
        
        # Interview data
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.evaluations: List[Dict] = []
        
        # Final feedback (populated at end)
        self.feedback: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """
        Convert session to dictionary format.
        
        Returns:
            Dictionary representation of session
        """
        return {
            "session_id": self.session_id,
            "job_title": self.job_title,
            "job_description": self.job_description,
            "cv_text": self.cv_text,
            "state": self.state.value,
            "questions": self.questions,
            "answers": self.answers,
            "evaluations": self.evaluations,
            "feedback": self.feedback
        }
    
    def get_qa_pairs(self) -> List[Dict]:
        """
        Get question-answer pairs with evaluations.
        
        Returns:
            List of Q&A dictionaries
        """
        pairs = []
        for i, (q, a) in enumerate(zip(self.questions, self.answers)):
            pair = {
                "question_number": i + 1,
                "question": q,
                "answer": a
            }
            if i < len(self.evaluations):
                pair["evaluation"] = self.evaluations[i]
            pairs.append(pair)
        return pairs
    
    def is_complete(self) -> bool:
        """
        Check if interview is complete.
        
        Returns:
            True if interview has ended
        """
        return self.state == InterviewState.ENDED