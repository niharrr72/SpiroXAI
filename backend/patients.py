"""
Patient storage: saves/loads patients per doctor.
"""

import json, os, uuid, datetime
from typing import List, Optional

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PATIENTS_FILE = os.path.join(BASE_DIR, "data", "patients.json")
os.makedirs(os.path.dirname(PATIENTS_FILE), exist_ok=True)

def _load() -> dict:
    if not os.path.exists(PATIENTS_FILE):
        return {}
    with open(PATIENTS_FILE) as f:
        return json.load(f)

def _save(data: dict):
    with open(PATIENTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def add_patient(doctor_email: str, name: str, sex: int, age: int, height: float, weight: float, race: str) -> dict:
    """Add a new patient for a doctor."""
    data = _load()
    if doctor_email not in data:
        data[doctor_email] = []
    
    patient_id = str(uuid.uuid4())
    patient = {
        "id": patient_id,
        "name": name,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "bmi": round(weight / ((height / 100) ** 2), 1),
        "race": race
    }
    data[doctor_email].append(patient)
    _save(data)
    return patient

def get_patients(doctor_email: str) -> List[dict]:
    """Return all patients for a doctor (newest first)."""
    data = _load()
    patients = data.get(doctor_email, [])
    # Sort by created_at descending
    return sorted(patients, key=lambda x: x.get("created_at", ""), reverse=True)

def get_patient(doctor_email: str, patient_id: str) -> Optional[dict]:
    """Return a single patient by ID."""
    for p in get_patients(doctor_email):
        if p["id"] == patient_id:
            return p
    return None
