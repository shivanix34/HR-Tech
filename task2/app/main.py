from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import os
from analysis_code import analyze_df

# Create directories
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Employee Analysis API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Employee Analysis API is running"}

@app.post("/analyze/")
async def analyze_survey(file: UploadFile = File(...)):
    """
    Process the survey.csv file from input folder and return analysis results
    """
    
    try:
        # Read and analyze the CSV
        df = pd.read_csv(file.file)
        result_df = analyze_df(df)
        
        # Save results in the specified format
        output_cols = ['Employee ID', 'Feedback', 'sentiment_label', 'sentiment_score', 'attrition_score', 'attrition_risk', 'suggestion']
        available_cols = [col for col in output_cols if col in result_df.columns]
        
        output_path = os.path.join(OUTPUT_DIR, 'analysis_result.csv')
        result_df[available_cols].to_csv(output_path, index=False)
        
        # Return the analyzed file
        return FileResponse(
            output_path, 
            media_type='text/csv', 
            filename='analysis_result.csv'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "message": "API is working properly"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)