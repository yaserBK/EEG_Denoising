import json
from pathlib import Path
from .config import PATH_TO_TUAR_JSON


def load_master() -> dict:
    with open(PATH_TO_TUAR_JSON / "master.json") as f:
        return json.load(f)


def load_patient(patient_id: str) -> dict:
    with open(PATH_TO_TUAR_JSON / f"{patient_id}.json") as f:
        return json.load(f)


def iter_patients(master: dict):
    for patient_id in master["Patients"]:
        yield patient_id, load_patient(patient_id)