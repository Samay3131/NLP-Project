from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from app.model import summarize_feedback

app = FastAPI(title="Feedback Summarizer API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Feedback Summarizer API 🚀"}

@app.post("/analyze/")
async def analyze_feedback(file: UploadFile):
    """
    Receives a CSV file with a 'feedback' column, summarizes each row, returns JSON.
    """
    try:
        # 1) Load the uploaded file as DataFrame:
        df = pd.read_csv(file.file)

        # 2) Validate the file has the required 'feedback' column:
        if 'feedback' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'feedback' column.")

        # 3) Summarize each row:
        summaries = []
        for text in df['feedback']:
            summary = summarize_feedback(text)
            summaries.append(summary)

        # 4) Add summaries to the dataframe:
        df['summary'] = summaries

        # 5) Return result as a list of dictionaries (JSON):
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
