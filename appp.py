from fastapi import FastAPI, HTTPException, Depends, File, UploadFile 
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import MriScan, Base  # Ensure 'models.py' exists

# Initialize FastAPI app
app = FastAPI()

# 1️⃣ **Database Connection Setup**
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your Heroku config vars.")

# Ensure compatibility with SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine & session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# 2️⃣ **Heroku-Compatible Storage Paths**
UPLOAD_DIR = "/app/mriT1/input"
FREESURFER_OUTPUT_DIR = "/app/mriT1/freesurfer_output"

# Ensure directories exist (only locally, not on Heroku)
if not os.getenv("DYNO"):  # Heroku doesn't allow writing to the filesystem
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FREESURFER_OUTPUT_DIR, exist_ok=True)

# 3️⃣ **Fix: Load DeepSeek Model Correctly**
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"  # Update with correct HF model

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")

# 4️⃣ **Database Dependency**
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 5️⃣ **API Endpoint: Test Server**
@app.get("/")
async def root():
    return {"message": "🚀 Alzheimer's Prediction API is running!"}

# 6️⃣ **API Endpoint: Upload MRI Scan**
@app.post("/upload/")
async def upload_mri_scan(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save uploaded file
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        
        # Save to DB (dummy example, modify as needed)
        new_scan = MriScan(filename=file.filename, file_path=file_location)
        db.add(new_scan)
        db.commit()
        
        return {"filename": file.filename, "status": "Uploaded Successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# 7️⃣ **API Endpoint: Predict using DeepSeek Model**
@app.post("/predict/")
async def predict(input_text: str):
    try:
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prediction": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")







