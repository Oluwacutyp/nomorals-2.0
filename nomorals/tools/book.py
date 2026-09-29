"""Book writing tools for AI-assisted content creation."""

from __future__ import annotations

from typing import Any


def book_create(topic: str = "", slug: str = "", chapters: int = 5, 
                research: bool = True, **kwargs) -> dict[str, Any]:
    """Create a new book project."""
    return {
        "topic": topic,
        "slug": slug or topic.lower().replace(" ", "-"),
        "chapters": chapters,
        "research": research,
        "status": "created"
    }


def book_run(topic: str = "", chapters: int = 5, words: int = 2000, 
             no_research: bool = False, research: bool = True, **kwargs) -> dict[str, Any]:
    """Run the book writing pipeline."""
    return {
        "topic": topic,
        "chapters": chapters,
        "words_per_chapter": words,
        "research": research and not no_research,
        "status": "completed"
    }


def book_write(topic: str = "", chapters: int = 5, words: int = 2000, **kwargs) -> dict[str, Any]:
    """Write a book."""
    return book_run(topic=topic, chapters=chapters, words=words, **kwargs)


def register(registry: Any) -> None:
    """Register book tools with the registry."""
    registry.register(
        "book_create",
        book_create,
        description="Create a new book project",
        capability="book",
        parameters={
            "topic": {"type": "string", "description": "Book topic"},
            "slug": {"type": "string", "description": "Book slug"},
            "chapters": {"type": "integer", "description": "Number of chapters"},
            "research": {"type": "boolean", "description": "Include research phase"}
        }
    )
    registry.register(
        "book_run",
        book_run,
        description="Run the book writing pipeline",
        capability="book",
        parameters={
            "topic": {"type": "string", "description": "Book topic"},
            "chapters": {"type": "integer", "description": "Number of chapters"},
            "words": {"type": "integer", "description": "Words per chapter"},
            "no_research": {"type": "boolean", "description": "Skip research phase"},
            "research": {"type": "boolean", "description": "Include research phase"}
        }
    )
    registry.register(
        "book_write",
        book_write,
        description="Write a book",
        capability="book",
        parameters={
            "topic": {"type": "string", "description": "Book topic"},
            "chapters": {"type": "integer", "description": "Number of chapters"},
            "words": {"type": "integer", "description": "Words per chapter"}
        }
    )
