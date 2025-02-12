from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import MriScan, Base  # Import MriScan model from models.py
import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sqlalchemy.orm import Session  # <-- This is missing in your code
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker






# Initialize FastAPI app
app = FastAPI()

# Database connection setup
DATABASE_URL = os.getenv("postgres://ug8j501cbdf1p:pb2217cb07c6d0335cd573f91e6a5d1847f66466a525e9d1fba05d53613d5e4c8@cf980tnnkgv1bp.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d4s1gt26vfkp6b")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Directories for MRI uploads and FreeSurfer outputs
UPLOAD_DIR = "/mnt/c/Users/vijay/Downloads/mriT1/input"
FREESURFER_OUTPUT_DIR = "/mnt/c/Users/vijay/Downloads/mriT1/freesurfer_output"
MODEL_PATH = "/mnt/c/Users/vijay/DeepSeek_Project/DeepSeek-Finetuned-Alzheimers"  # Fine-tuned model path

# Ensure the directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FREESURFER_OUTPUT_DIR, exist_ok=True)

# Load the DeepSeek model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to run DeepSeek prediction
def run_deepseek_prediction(prompt: str):
    """Generate a prediction using the fine-tuned DeepSeek model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the input model for predictions
class MRIInput(BaseModel):
    patient_id: str
    biomarker_data: dict
    structured_prompt: str

@app.post("/predict")
def predict_mri(input_data: MRIInput, db: Session = Depends(get_db)):
    """
    Use DeepSeek AI to analyze MRI biomarkers and predict Alzheimer's risk.
    Store the prediction and biomarkers in the database.
    """
    structured_text = f"""
    Patient {input_data.patient_id} MRI analysis:
    Biomarker Data: {input_data.biomarker_data}
    MMSE Score Prediction:
    """.strip()

    prediction_text = run_deepseek_prediction(structured_text)

    # Store the prediction in the database
    mri_scan = MriScan(
        patient_id=input_data.patient_id,
        file_name=f"{input_data.patient_id}_MRI",
        status="completed",
        prediction=prediction_text,
        biomarkers=input_data.biomarker_data
    )
    db.add(mri_scan)
    db.commit()
    db.refresh(mri_scan)

    return {"patient_id": input_data.patient_id, "prediction": prediction_text}

@app.post("/process_mri/")  # Updated to store the MRI scan in DB
async def process_mri(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload an MRI scan (.nii) → Get **instant DeepSeek AI prediction**.
    FreeSurfer runs in the background for refined biomarkers.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    subject_id = file.filename.split(".")[0]

    # Start FreeSurfer in the background (this is just a placeholder)
    freesurfer_cmd = f"recon-all -s {subject_id} -i {file_path} -all"
    subprocess.Popen(freesurfer_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Store MRI scan details in the database
    new_scan = MriScan(patient_id=subject_id, file_name=file.filename, status="processing")
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    # Instant DeepSeek AI prediction
    ai_prediction = run_deepseek_prediction(f"Immediate AI scan analysis for {subject_id}")

    # Update the database with the AI prediction
    new_scan.prediction = ai_prediction
    db.commit()

    return {
        "message": f"AI prediction is available immediately for {subject_id}. FreeSurfer is processing in the background.",
        "subject_id": subject_id,
        "status": new_scan.status,
        "instant_prediction": ai_prediction
    }

@app.get("/check_status/{subject_id}")
def check_status(subject_id: str, db: Session = Depends(get_db)):
    """
    Check the processing status of an MRI scan and prediction from the database.
    """
    mri_scan = db.query(MriScan).filter(MriScan.patient_id == subject_id).first()

    if not mri_scan:
        raise HTTPException(status_code=404, detail="Subject ID not found")

    return {
        "subject_id": subject_id,
        "status": mri_scan.status,
        "prediction": mri_scan.prediction,
        "biomarkers": mri_scan.biomarkers
    }

@app.get("/extract_biomarkers/{subject_id}")
def extract_biomarkers(subject_id: str, db: Session = Depends(get_db)):
    """
    Extract biomarkers from FreeSurfer `aseg.stats` after MRI processing.
    """
    stats_file = f"{FREESURFER_OUTPUT_DIR}/{subject_id}/stats/aseg.stats"

    if not os.path.exists(stats_file):
        raise HTTPException(status_code=404, detail="Biomarker stats not found. MRI may still be processing.")

    biomarkers = {}
    keywords = ["Hippocampus", "Ventricle", "Thalamus", "Cortex", "Amygdala"]

    with open(stats_file, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) > 4 and any(keyword in parts[4] for keyword in keywords):
                region = parts[4]
                volume = float(parts[3])
                biomarkers[region] = volume

    # Update DeepSeek Prediction with Biomarkers in the database
    updated_prediction = run_deepseek_prediction(f"Updated analysis for {subject_id} with biomarkers: {biomarkers}")

    mri_scan = db.query(MriScan).filter(MriScan.patient_id == subject_id).first()
    if not mri_scan:
        raise HTTPException(status_code=404, detail="Subject ID not found")

    mri_scan.biomarkers = biomarkers
    mri_scan.prediction = updated_prediction
    db.commit()

    return {
        "subject_id": subject_id,
        "biomarkers": biomarkers,
        "updated_prediction": updated_prediction
    }

