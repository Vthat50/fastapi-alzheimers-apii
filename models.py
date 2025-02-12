from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

# Define the Base class for our models
Base = declarative_base()

class MriScan(Base):
    __tablename__ = "mri_scans"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    file_name = Column(String)
    status = Column(String, default="processing")
    prediction = Column(String, default="processing...")
    biomarkers = Column(JSON, default={})  # Store biomarkers as a JSON object

    # Additional columns and relationships can be added here if needed
