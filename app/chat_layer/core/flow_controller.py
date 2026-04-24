from app.chat_layer.models.interview import InterviewState

def next_state(current_state):
    if current_state == InterviewState.INTRO:
        return InterviewState.TECHNICAL
    elif current_state == InterviewState.TECHNICAL:
        return InterviewState.COMMUNICATION
    else:
        return InterviewState.ENDED
