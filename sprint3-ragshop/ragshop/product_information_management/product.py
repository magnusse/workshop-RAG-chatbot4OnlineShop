from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Product:
    """Master product as owned by the PIM bounded context.
    Sprint 3 schema: just enough to describe + index a product."""

    id: str
    name: str
    category: str
    description: str
    price: str
    compatibility: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """DTO used when pushing across the boundary to Sales Consultance."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "price": self.price,
            "compatibility": list(self.compatibility),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Product":
        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            description=data["description"],
            price=data["price"],
            compatibility=list(data.get("compatibility", [])),
        )
