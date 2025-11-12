from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = FastAPI(title="erg.ai", description="Analyze rowing ergometer data with AI insights")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve front-end
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Core analysis function ---
def analyze_erg_from_bytes(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
    'time (s)': 'time',
    'time (seconds)': 'time',
    'pace (sec/500m)': 'pace',
    'pace (seconds)': 'pace',
    'watts': 'watts',
    'stroke rate': 'stroke_rate',
    'heart rate': 'heart_rate'
}, inplace=True)


    avg_split = round(float(df["pace"].mean()), 2)
    avg_power = round(float(df["watts"].mean()), 1)
    consistency = round(float(df["watts"].std()), 2)
    drift = round(float(df["watts"].iloc[-1] - df["watts"].iloc[0]), 2)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["watts"], label="Watts", color="orange", linewidth=2)
    plt.plot(df["time"], df["pace"], label="Pace", color="blue", linestyle="--")
    if "heart_rate" in df.columns:
        plt.plot(df["time"], df["heart_rate"], label="Heart Rate", color="red", alpha=0.6)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Metrics")
    plt.title("Erg Workout Analysis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "avg_split": avg_split,
        "avg_power": avg_power,
        "consistency": consistency,
        "drift": drift,
        "plot": img_base64
    }


# --- Single correct POST endpoint ---
@app.post("/analyze")
async def analyze_workout(file: UploadFile = File(...)):
    contents = await file.read()
    return analyze_erg_from_bytes(contents)
