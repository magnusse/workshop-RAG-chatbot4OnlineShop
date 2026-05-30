from enum import Enum


class Intent(str, Enum):
    PRODUCT_INFO = "PRODUCT_INFO"
    OFFER_REQUEST = "OFFER_REQUEST"
    SMALLTALK = "SMALLTALK"

    @classmethod
    def parse(cls, raw: str) -> "Intent":
        normalized = (raw or "").strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.SMALLTALK
