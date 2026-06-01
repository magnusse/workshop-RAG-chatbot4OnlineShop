from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Product:
    """Master product as owned by the PIM bounded context.

    Sprint 5 grows the schema: now carries source, upddate, delflag,
    prodcatversion — these are needed for the runtime update path and
    for retrieval-side filtering (Domain Story Step 5 refined)."""

    id: str
    name: str
    category: str
    description: str
    price: str
    source: str
    upddate: str
    delflag: bool
    prodcatversion: str
    compatibility: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "price": self.price,
            "source": self.source,
            "upddate": self.upddate,
            "delflag": self.delflag,
            "prodcatversion": self.prodcatversion,
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
            source=data.get("source", ""),
            upddate=data["upddate"],
            delflag=data["delflag"],
            prodcatversion=data["prodcatversion"],
            compatibility=list(data.get("compatibility", [])),
        )
