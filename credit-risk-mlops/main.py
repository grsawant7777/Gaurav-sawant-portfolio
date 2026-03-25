import os
import json
from datetime import datetime
from typing import List, Optional, Any
from io import StringIO # <-- FIX 1: Import StringIO from the standard 'io' library

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import numpy as np 

# === CONFIGURATION ===
RESULTS_DIR = "results"
SINGLE_BATCH_DIR = os.path.join(RESULTS_DIR, "single_batch")
CSV_UPLOADS_DIR = os.path.join(RESULTS_DIR, "csv_uploads")
MODEL_FILE = 'models/credit_risk_model.pkl'

# Ensure output directories exist
os.makedirs(SINGLE_BATCH_DIR, exist_ok=True)
os.makedirs(CSV_UPLOADS_DIR, exist_ok=True)

# === DATA MODELS (Pydantic) ===

class LoanApplication(BaseModel):
    """
    Input validation model for a single loan application.
    """
    person_age: int = Field(..., ge=18)
    person_income: int = Field(..., ge=0)
    person_home_ownership: str
    
    person_emp_length: Optional[float] = Field(None, ge=0) 
    
    loan_intent: str
    loan_grade: str
    loan_amnt: int = Field(..., ge=100)
    
    loan_int_rate: Optional[float] = Field(None, gt=0.0)
    
    loan_percent_income: float = Field(..., gt=0.0, le=1.0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0)

class BatchLoanApplications(BaseModel):
    """Input validation model for a batch of loan applications."""
    applications: List[LoanApplication]


# === RISK ENGINE CLASS (Encapsulating Model Logic) ===

class RiskEngine:
    """Handles model loading, prediction, and result categorization."""
    
    def __init__(self, model_path: str):
        self.full_pipeline = None
        self.expected_features = []
        self._load_artifacts(model_path)
    
    def _load_artifacts(self, model_path: str):
        """Loads the saved scikit-learn pipeline and feature list."""
        if os.path.exists(model_path):
            try:
                artifact = joblib.load(model_path)
                self.full_pipeline = artifact['pipeline']
                self.expected_features = artifact['features']
                print("✅ Model artifacts loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading model artifacts: {e}. Run train_model.py first.")
                self.full_pipeline = None
                
    def is_ready(self) -> bool:
        """Check if the model pipeline is loaded."""
        return self.full_pipeline is not None
        
    def _categorize_risk(self, p: float) -> str:
        """Determines the risk category based on default probability."""
        if p >= 0.4:
            return "HIGH_RISK"
        elif p >= 0.2:
            return "MODERATE_RISK"
        else:
            return "LOW_RISK"
            
    def predict(self, data: dict) -> dict:
        """Runs a single prediction on input data."""
        if not self.is_ready():
            raise HTTPException(503, "Model unavailable for prediction.")
            
        df = pd.DataFrame([data], columns=self.expected_features)
        
        proba = self.full_pipeline.predict_proba(df)[:, 1][0]
        
        # Convert numpy.float to standard Python float for JSON serialization
        proba_native = float(proba) 
        
        return {
            "probability_of_default": round(proba_native * 100, 2),
            "risk_category": self._categorize_risk(proba_native)
        }

# Initialize the Risk Engine globally
engine = RiskEngine(MODEL_FILE)


# === FASTAPI UTILITIES ===

def save_prediction_result(input_data: Any, results: List[dict], is_batch: bool):
    """Saves single or batch prediction results to a JSON file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if is_batch:
        filename = f"{timestamp}_batch.json"
        path = os.path.join(SINGLE_BATCH_DIR, filename)
        output_data = {"batch_size": len(results), "predictions": results}
    else:
        filename = f"{timestamp}_single.json"
        path = os.path.join(SINGLE_BATCH_DIR, filename)
        output_data = {"input": input_data, **results[0]}
        
    with open(path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data 


# === FASTAPI APPLICATION SETUP ===

app = FastAPI(title="CreditGuard | AI Risk Engine", version="2.1")


# === ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serves the frontend UI."""
    # FIX: Explicitly specify UTF-8 encoding to avoid UnicodeDecodeError on Windows/different systems.
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/results/single_batch/{filename}")
async def download_single_batch(filename: str):
    """Allows downloading of single/batch JSON results."""
    path = os.path.join(SINGLE_BATCH_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    raise HTTPException(404, "File not found.")


@app.post("/predict")
async def predict_risk(app: LoanApplication):
    """Endpoint for single loan application prediction."""
    try:
        input_data = app.model_dump() 
        result = engine.predict(input_data)
        
        save_prediction_result(input_data, [result], is_batch=False)
        return {**result, "input": input_data}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Single prediction failed: {e}")
        raise HTTPException(400, f"Prediction failed: {e}")


@app.post("/predict_batch")
async def predict_batch_risk(batch: BatchLoanApplications):
    """Endpoint for batch loan application prediction."""
    if not batch.applications: raise HTTPException(400, "Empty batch")
    
    try:
        inputs = [a.model_dump() for a in batch.applications]
        results = []
        
        for inp in inputs:
            result = engine.predict(inp)
            results.append({"input": inp, **result})

        saved_output = save_prediction_result(None, results, is_batch=True)
        
        return saved_output
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        raise HTTPException(400, f"Batch failed: {e}")


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Endpoint for predicting risk from an uploaded CSV file."""
    if not file.filename.endswith('.csv'): raise HTTPException(400, "CSV files only.")
    if not engine.is_ready(): raise HTTPException(503, "Model unavailable")
    
    try:
        # Read file content
        content = await file.read()
        
        df = pd.read_csv(StringIO(content.decode())) 
        
        missing = [c for c in engine.expected_features if c not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing required columns in CSV: {missing}")
            
        df_input = df[engine.expected_features]
        probs = engine.full_pipeline.predict_proba(df_input)[:, 1]
        
        output_rows = []
        for idx, (row, p) in enumerate(zip(df_input.iterrows(), probs)):
            p_native = float(p) # FIX: Convert NumPy float for serialization
            
            output_rows.append({
                "person_age": row[1]['person_age'], 
                "loan_amnt": row[1]['loan_amnt'],
                "probability_of_default": round(p_native * 100, 2),
                "risk_category": engine._categorize_risk(p_native)
            })

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_report.csv"
        output_path = os.path.join(CSV_UPLOADS_DIR, filename)
        
        pd.DataFrame(output_rows).to_csv(output_path, index=False)
        
        return FileResponse(output_path, filename="credit_risk_report.csv")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"CSV processing failed: {e}")
        raise HTTPException(400, f"CSV processing failed: {e}")


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)