from dataclasses import dataclass


@dataclass(frozen=True)
class CustomerId:
    value: str


@dataclass(frozen=True)
class Customer:
    id: CustomerId
