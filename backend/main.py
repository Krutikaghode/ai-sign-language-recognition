from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI(
    title="AI Sign Language Recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LandmarkRequest(BaseModel):
    landmarks: List[float]

class LandmarkResponse(BaseModel):
    gesture: str
    confidence: float
    message: str

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "service": "sign-language-recognition-backend"}

@app.post("/api/v1/predict/landmarks", response_model=LandmarkResponse)
async def predict_landmarks(request: LandmarkRequest):
    if len(request.landmarks) != 126:
        return LandmarkResponse(
            gesture="UNKNOWN",
            confidence=0.0,
            message="Invalid landmark vector length. Expected 126 floats."
        )
    return LandmarkResponse(
        gesture="HELLO",
        confidence=0.95,
        message="Prediction successful"
    )

@app.get("/api/v1/history")
async def get_history(limit: int = 50):
    history_data = [
        {"id": 1, "gesture": "HELLO", "confidence": 0.96, "timestamp": "2026-11-02 10:00:00"},
        {"id": 2, "gesture": "THANK_YOU", "confidence": 0.92, "timestamp": "2026-11-02 10:01:15"},
        {"id": 3, "gesture": "PLEASE", "confidence": 0.89, "timestamp": "2026-11-02 10:02:30"}
    ]
    return {"history": history_data[:limit]}

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            landmarks = payload.get("landmarks", [])
            if len(landmarks) == 126:
                response = {
                    "gesture": "HELLO",
                    "confidence": 0.95,
                    "status": "success"
                }
            else:
                response = {
                    "gesture": "UNKNOWN",
                    "confidence": 0.0,
                    "status": "error",
                    "message": "Landmark vector must be 126 dimensions"
                }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
