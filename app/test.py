from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import os
import logging
import base64
from collections import deque
import asyncio

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enable CORS with WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load environment variables
load_dotenv()
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_API_ENDPOINT = os.getenv("FACEPP_API_ENDPOINT")

# Ensure that the API key, secret, and endpoint are set
if not FACEPP_API_KEY or not FACEPP_API_SECRET or not FACEPP_API_ENDPOINT:
    raise ValueError("One or more environment variables for Face++ API are not set.")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# A deque to store the last 10 emotion readings
emotion_history = deque(maxlen=10)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading main page: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading page</h1><p>{str(e)}</p>", status_code=500)

@app.get("/favicon.ico", response_class=HTMLResponse)
async def favicon():
    return HTMLResponse(content="<link rel='icon' href='/static/webcam_favicon_orange_black.ico' type='image/x-icon'>", status_code=200)

def extract_dominant_attribute(attribute_dict):
    return max(attribute_dict, key=attribute_dict.get)

def calculate_most_intense_emotion():
    combined_emotions = {
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "happiness": 0,
        "sadness": 0,
        "surprise": 0
    }

    # Sum up all the intensities for each emotion across the last 10 readings
    for emotions in emotion_history:
        for emotion, intensity in emotions.items():
            combined_emotions[emotion] += intensity

    # Find the most intense emotion
    most_intense_emotion = max(combined_emotions, key=combined_emotions.get)
    most_intense_value = combined_emotions[most_intense_emotion]

    if most_intense_value > 0:
        logging.info(f"Most intense emotion: {most_intense_emotion} with intensity {most_intense_value}")
        return f"You feel {most_intense_emotion}"
    else:
        logging.info("No prevailing emotion detected")
        return "You feel in progress..."

async def process_emotion_image(image_bytes):
    try:
        files = {'image_file': image_bytes}
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'return_attributes': 'emotion,headpose,eyestatus'
        }

        response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
        response.raise_for_status()

        faces = response.json().get('faces', [])
        if faces:
            face_attributes = faces[0]['attributes']
            emotions = face_attributes['emotion']
            emotion_history.append(emotions)
            return {
                "emotions": emotions,
                "head_pose": face_attributes['headpose'],
                "eye_status": face_attributes['eyestatus'],
                "dominant_emotion": extract_dominant_attribute(emotions),
                "additional_emotion": calculate_most_intense_emotion()
            }
        else:
            return {"error": "No face detected"}
    except Exception as e:
        logging.error(f"Face++ API error: {str(e)}")
        return {"error": str(e)}

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established")

    image_bytes = None

    while True:
        try:
            # Check for new image data
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    data = await websocket.receive_text()
                    image_data = data.split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                except Exception as e:
                    logging.error(f"Error receiving image data: {e}")

            # Process image every 2 seconds
            if image_bytes:
                results = await process_emotion_image(image_bytes)
                await websocket.send_json(results)
                await asyncio.sleep(2)  # Wait for 2 seconds before the next measurement

        except WebSocketDisconnect:
            logging.info("Client disconnected")
            break
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")
            await websocket.send_json({"error": "Failed to process image."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
