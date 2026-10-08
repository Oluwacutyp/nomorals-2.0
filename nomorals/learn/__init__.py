"""Devon as tutor — Socratic tutoring, photo input, mistake notebook.

Build-map #46. ``tutor.py`` is the dialogue engine (socratic vs direct
modes, mastery model, answer-guard); ``flashcards.py`` turns wrong
answers into spaced-repetition cards.
"""

from .flashcards import MistakeNotebook, card_from_mistake
from .tutor import (
    MasteryModel,
    SocraticEngine,
    TutorError,
    TutorSession,
    TutorTurn,
    end_session,
    get_notebook,
    get_session,
    set_session,
    tutor_from_files,
    tutor_from_photo,
)

__all__ = [
    "MasteryModel",
    "MistakeNotebook",
    "SocraticEngine",
    "TutorError",
    "TutorSession",
    "TutorTurn",
    "card_from_mistake",
    "end_session",
    "get_notebook",
    "get_session",
    "set_session",
    "tutor_from_files",
    "tutor_from_photo",
]
