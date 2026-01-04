from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
from datetime import date, datetime
import shutil
import io
from PIL import Image

from model_utils import get_model
from qc_utils import check_quality

app = FastAPI(title="Retinal Age Gap Research App", version="1.0.0")

# Mount static files
# Mount static files
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model on startup
@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    laterality: str = Form(...)
):
    if laterality not in ['OD', 'OS']:
        raise HTTPException(status_code=400, detail="Invalid laterality. Must be OD or OS.")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # QC Check
        qc_result = check_quality(img)
        
        # Prediction
        model = get_model()
        predicted_age = model.predict(img, laterality)
        
        return {
            "predicted_age": predicted_age,
            "qc_status": qc_result['status'],
            "qc_metrics": qc_result['metrics'],
            "qc_reasons": qc_result['reasons'],
            "server_date": date.today().isoformat(),
            "model_version": "JOIR_Swin_20220903",
            "app_version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_gap")
async def calculate_gap(
    dob: str = Form(...),
    predicted_age: float = Form(...)
):
    try:
        dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
        today = date.today()
        
        # Calculate chronological age
        chrono_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        
        # Calculate gap
        gap = predicted_age - chrono_age
        
        return {
            "chronological_age": chrono_age,
            "retinal_age_gap": round(gap, 1)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid DOB format. Use YYYY-MM-DD.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
