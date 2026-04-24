from enum import Enum

class InterviewState(Enum):
    INTRO = "intro"
    TECHNICAL = "technical"
    FOLLOW_UP = "follow_up"
    BEHAVIORAL = "behavioral"
    ENDED = "ended"
