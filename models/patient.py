from dataclasses import dataclass, field

@dataclass
class Patient:
    age: int
    sex: str
    acne_severity: str

    acne_duration_years: float | None = None
    hormonal_factors: list[str] = field(default_factory=list)
    environmental_factors: list[str] = field(default_factory=list)