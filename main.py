from models.patient import Patient

parient = Patient(
    age=24,
    sex="female",
    acne_severity= "moderate",
    acne_duration_years= 5.0,
    hormonal_factors=["hormonal_fluctuations"],
    environmental_factors=["stress"]
)

print(parient)