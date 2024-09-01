from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import os
import logging
import base64
from collections import deque, Counter
import uvicorn

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

# A deque to store the last 5 emotion readings
emotion_history = deque(maxlen=5)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    try:
        with open("static/final.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading main page: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading page</h1><p>{str(e)}</p>", status_code=500)

def adjust_emotions_for_head_pose_and_eyes(emotions, head_pose, eye_status):
    pitch = head_pose['pitch_angle']
    yaw = head_pose['yaw_angle']

    # Adjust emotions based on pitch
    if pitch < -10:  # Significant downward tilt
        emotions['happiness'] *= 0.7  # Reduce happiness intensity
    elif pitch > 10:  # Significant upward tilt
        emotions['surprise'] *= 1.3  # Enhance surprise
        emotions['happiness'] *= 1.2  # Enhance happiness

    # Adjust emotions based on yaw
    if abs(yaw) > 20:  # Large yaw (side-to-side movement)
        emotions['happiness'] *= 0.8  # Reduce happiness
        emotions['disgust'] *= 1.2  # Increase negative emotion like disgust

    # Adjust emotions based on eye status
    eye_openness = (eye_status['left_eye_status']['no_glass_eye_open'] + eye_status['right_eye_status']['no_glass_eye_open']) / 2
    if eye_openness > 0.8:  # Eyes wide open
        emotions['surprise'] *= 1.4  # Increase surprise
        emotions['fear'] *= 1.3  # Increase fear
    elif eye_openness < 0.3:  # Eyes closed or nearly closed
        emotions['neutral'] *= 1.2  # Increase neutrality
        emotions['disgust'] *= 1.1  # Increase negative emotions like disgust or suspicion

    return emotions

def extract_dominant_attribute(attribute_dict):
    return max(attribute_dict, key=attribute_dict.get)

def calculate_most_intense_emotion(emotions, head_pose, eye_status):
    combined_emotions = Counter()

    for emotion_reading in emotion_history:
        adjusted_emotions = adjust_emotions_for_head_pose_and_eyes(emotion_reading, head_pose, eye_status)
        combined_emotions.update(adjusted_emotions)

    most_intense_emotion = combined_emotions.most_common(1)
    if most_intense_emotion:
        emotion, intensity = most_intense_emotion[0]
        logging.info(f"Most intense emotion: {emotion} with intensity {intensity}")
        return emotion
    else:
        logging.info("No prevailing emotion detected")
        return None

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established")
    while True:
        try:
            data = await websocket.receive_text()
            image_data = data.split(",")[1]
            image_bytes = base64.b64decode(image_data)

            results = {
                "Face++": {
                    "emotions": None,
                    "head_pose": None,
                    "eye_status": None,
                    "dominant_emotion": None,
                    "most_prevalent_emotion": "Running...",
                    "error": None
                }
            }

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
                    head_pose = face_attributes['headpose']
                    eye_status = face_attributes['eyestatus']

                    emotion_history.append(emotions)

                    results["Face++"]["emotions"] = emotions
                    results["Face++"]["head_pose"] = head_pose
                    results["Face++"]["eye_status"] = eye_status
                    results["Face++"]["dominant_emotion"] = extract_dominant_attribute(emotions)

                    if len(emotion_history) == 5:
                        most_intense_emotion = calculate_most_intense_emotion(emotions, head_pose, eye_status)
                        if most_intense_emotion:
                            results["Face++"]["most_prevalent_emotion"] = f"Most intense: {most_intense_emotion}"
                        emotion_history.clear()
                    
                    logging.info(f"Face++ API response: {response.json()}")
                else:
                    results["Face++"]["error"] = "No face detected"
            except Exception as e:
                logging.error(f"Face++ API error: {str(e)}")
                results["Face++"]["error"] = str(e)
            
            await websocket.send_json(results)
        except WebSocketDisconnect:
            logging.info("Client disconnected")
            break
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")
            await websocket.send_json({"error": "Failed to process image."})
