"""
FastAPI main application — Lung Disease Diagnostic System
"""

import os, json
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

import auth, history, report, patients
from ml_pipeline import get_predictor, CLASS_NAMES, FEATURE_NAMES

app = FastAPI(title="SpiroXAI — Lung Disease Diagnostic API", version="1.0.0")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ── Pydantic models ────────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class PatientRequest(BaseModel):
    name: str
    sex: int
    age: int
    height: float
    weight: float
    race: str

class PredictRequest(BaseModel):
    patient_id: str
    patient_name: str
    # 15 raw features
    Sex: float                                  # 0=Female, 1=Male
    Age: float
    Weight: float
    Height: float
    BMI: float
    Baseline_PEF_Ls: float
    Baseline_FEF2575_Ls: float
    Baseline_Extrapolated_Volume: float
    Baseline_Forced_Expiratory_Time: float
    Baseline_Number_Acceptable_Curves: float
    Race_Black: float = 0.0
    Race_Mexican_American: float = 0.0
    Race_Other_hispanic: float = 0.0
    Race_Other_race_including_multi_racial: float = 0.0
    Race_White: float = 1.0

# ── Auth dependency ────────────────────────────────────────────────────────────
def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ", 1)[1]
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return payload

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES, "n_features": len(FEATURE_NAMES)}

@app.post("/auth/signup")
def signup(req: SignupRequest):
    try:
        result = auth.signup(req.email, req.password, req.name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
def login(req: LoginRequest):
    try:
        result = auth.login(req.email, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.get("/patients")
def get_patients_endpoint(user: dict = Depends(get_current_user)):
    return patients.get_patients(user["sub"])

@app.post("/patients")
def add_patient_endpoint(req: PatientRequest, user: dict = Depends(get_current_user)):
    return patients.add_patient(
        doctor_email=user["sub"],
        name=req.name,
        sex=req.sex,
        age=req.age,
        height=req.height,
        weight=req.weight,
        race=req.race
    )

@app.post("/predict")
def predict(req: PredictRequest, user: dict = Depends(get_current_user)):
    predictor = get_predictor()

    # Build raw input dict (with Race column fix for space in name)
    raw = req.dict()
    patient_id = raw.pop("patient_id")
    patient_name = raw.pop("patient_name")
    raw["Race_Mexican American"] = raw.pop("Race_Mexican_American", 0.0)
    raw["Race_Other hispanic"]   = raw.pop("Race_Other_hispanic", 0.0)
    raw["Race_Other race, including multi-racial"] = raw.pop("Race_Other_race_including_multi_racial", 0.0)

    # Run prediction
    pred_result = predictor.predict(raw)
    pred_idx  = pred_result["predicted_class"]
    proba_arr = pred_result["probabilities"]

    proba_dict = {CLASS_NAMES[i]: round(float(proba_arr[i]), 4) for i in range(len(CLASS_NAMES))}
    confidence_pct = round(float(proba_arr[pred_idx]) * 100, 1)

    # Explanation
    explanation = predictor.explain(raw, pred_result)

    response_payload = {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence_pct": confidence_pct,
        "probabilities": proba_dict,
        "explanation": explanation,
        "is_heuristic": pred_result.get("is_heuristic", False),
    }

    # Save to history
    record_id = history.add_record(
        doctor_email=user["sub"],
        patient_id=patient_id,
        patient_name=patient_name,
        patient_input=raw,
        prediction=response_payload,
    )
    response_payload["record_id"] = record_id

    return response_payload

@app.get("/history")
def get_history(user: dict = Depends(get_current_user)):
    records = history.get_records(user["sub"])
    # Trim for list view
    return [
        {
            "id": r["id"],
            "patient_name": r.get("patient_name", "Unknown Patient"),
            "timestamp": r["timestamp"],
            "prediction": r["prediction"].get("prediction"),
            "confidence_pct": r["prediction"].get("confidence_pct"),
        }
        for r in records
    ]

@app.get("/history/{record_id}")
def get_single_record(record_id: str, user: dict = Depends(get_current_user)):
    record = history.get_record(user["sub"], record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

@app.delete("/history/{record_id}")
def delete_history_record(record_id: str, user: dict = Depends(get_current_user)):
    success = history.delete_record(user["sub"], record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "success", "id": record_id}

@app.get("/report/{record_id}")
def download_report(record_id: str, user: dict = Depends(get_current_user)):
    record = history.get_record(user["sub"], record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    pdf_bytes = report.generate_report_pdf(record, doctor_name=user.get("name", "Doctor"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="lung_report_{record_id[:8]}.pdf"'},
    )

@app.get("/features")
def get_features():
    """Return feature names and metadata for the frontend form."""
    return {
        "raw_features": [
            {"name": "Sex",                                "label": "Sex",                          "type": "select",   "options": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}]},
            {"name": "Age",                                "label": "Age (years)",                  "type": "number",   "min": 5,   "max": 100, "step": 1},
            {"name": "Weight",                             "label": "Weight (kg)",                  "type": "number",   "min": 10,  "max": 200, "step": 0.1},
            {"name": "Height",                             "label": "Height (cm)",                   "type": "number",   "min": 50,  "max": 250, "step": 0.1},
            {"name": "BMI",                                "label": "BMI (kg/m²)",                  "type": "number",   "min": 10,  "max": 60,  "step": 0.1},
            {"name": "Baseline_PEF_Ls",                   "label": "PEF (L/s)",                    "type": "number",   "min": 0,   "max": 20,  "step": 0.01},
            {"name": "Baseline_FEF2575_Ls",               "label": "FEF 25–75% (L/s)",             "type": "number",   "min": 0,   "max": 10,  "step": 0.01},
            {"name": "Baseline_Extrapolated_Volume",      "label": "Extrapolated Volume (L)",       "type": "number",   "min": 0,   "max": 5,   "step": 0.001},
            {"name": "Baseline_Forced_Expiratory_Time",   "label": "Forced Expiratory Time (s)",   "type": "number",   "min": 0,   "max": 20,  "step": 0.01},
            {"name": "Baseline_Number_Acceptable_Curves", "label": "Acceptable Curves (#)",         "type": "number",   "min": 0,   "max": 10,  "step": 1},
        ],
        "race_features": [
            {"name": "Race_Black",                         "label": "Black"},
            {"name": "Race_Mexican_American",              "label": "Mexican American"},
            {"name": "Race_Other_hispanic",                "label": "Other Hispanic"},
            {"name": "Race_Other_race_including_multi_racial", "label": "Other / Multi-racial"},
            {"name": "Race_White",                         "label": "White"},
        ],
        "class_names": CLASS_NAMES,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
