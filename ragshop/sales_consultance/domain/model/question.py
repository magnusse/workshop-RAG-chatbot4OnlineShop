from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Question text must not be empty")


@dataclass(frozen=True)
class Answer:
    text: str
