from transformers import pipeline
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

if not FACEPP_API_KEY or not FACEPP_API_SECRET or not FACEPP_API_ENDPOINT:
    raise ValueError("One or more environment variables for Face++ API are not set.")

# Initialize a compact LLM pipeline (distilgpt2 for text generation)
try:
    llm_pipeline = pipeline('text-generation', model='distilgpt2')
    logging.info("LLM distilgpt2 successfully initialized.")
except Exception as e:
    logging.error(f"Error initializing LLM: {str(e)}")


# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

emotion_history = deque(maxlen=5)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    try:
        with open("static/test.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading main page: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading page</h1><p>{str(e)}</p>", status_code=500)

def extract_dominant_attribute(attribute_dict):
    return max(attribute_dict, key=attribute_dict.get)

def calculate_most_intense_emotion():
    combined_emotions = Counter()

    for emotions in emotion_history:
        combined_emotions.update(emotions)

    most_intense_emotion = combined_emotions.most_common(1)
    if most_intense_emotion:
        emotion, intensity = most_intense_emotion[0]
        logging.info(f"Most intense emotion: {emotion} with intensity {intensity}")
        return emotion
    else:
        logging.info("No prevailing emotion detected")
        return None

def adjust_emotions_by_head_pose_and_eye_status(emotions, head_pose, eye_status):
    pitch = head_pose['pitch_angle']
    yaw = head_pose['yaw_angle']

    if pitch < 0:
        emotions['happiness'] *= 0.8  
    elif pitch > 15:  
        emotions['surprise'] *= 1.2  
        emotions['happiness'] *= 1.2

    if abs(yaw) > 15:
        emotions['happiness'] *= 0.8
        emotions['disgust'] *= 1.2

    left_eye_open = eye_status['left_eye_status']['no_glass_eye_open']
    right_eye_open = eye_status['right_eye_status']['no_glass_eye_open']

    if left_eye_open > 0.8 and right_eye_open > 0.8:  
        emotions['surprise'] *= 1.2
        emotions['fear'] *= 1.2  
    elif left_eye_open < 0.4 and right_eye_open < 0.4:  
        emotions['tiredness'] = 1.0  
        emotions['boredom'] = 1.0  

    return emotions
@app.post("/generate-response")
async def generate_llm_response(prompt: str):
    try:
        # Ensure that llm_pipeline is correctly handled
        llm_output = llm_pipeline(prompt, max_length=100, num_return_sequences=1)

        # Ensure we handle potential generator types
        if isinstance(llm_output, list) and len(llm_output) > 0:
            if isinstance(llm_output[0], dict) and 'generated_text' in llm_output[0]:
                generated_text = llm_output[0]['generated_text']
                return {"generated_text": generated_text}
            else:
                return {"error": "Unexpected pipeline output format"}
        else:
            return {"error": "No valid response from the model"}
    except Exception as e:
        logging.error(f"Error generating response with LLM: {str(e)}")
        return {"error": "Failed to generate response with the LLM"}



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

            if not FACEPP_API_ENDPOINT:
                results["Face++"]["error"] = "Face++ API endpoint not configured"
                await websocket.send_json(results)
                return

            try:
                # Send request to Face++ API for emotion detection
                response = requests.post(
                    FACEPP_API_ENDPOINT,
                    data={"api_key": FACEPP_API_KEY, "api_secret": FACEPP_API_SECRET, "return_attributes": "emotion,headpose,eyestatus"},
                    files={"image_file": ("image.jpg", image_bytes)}
                )
                facepp_data = response.json()

                if 'error_message' in facepp_data:
                    results["Face++"]["error"] = facepp_data['error_message']
                else:
                    face_attributes = facepp_data['faces'][0]['attributes']
                    emotions = face_attributes['emotion']
                    head_pose = face_attributes['headpose']
                    eye_status = face_attributes['eyestatus']

                    adjusted_emotions = adjust_emotions_by_head_pose_and_eye_status(emotions, head_pose, eye_status)

                    dominant_emotion = extract_dominant_attribute(adjusted_emotions)

                    emotion_history.append(adjusted_emotions)

                    most_intense_emotion = calculate_most_intense_emotion()
                    if most_intense_emotion:
                        most_prevalent_emotion = f'You feel {most_intense_emotion} based on recent emotions'
                    else:
                        most_prevalent_emotion = 'Running...'

                    results["Face++"]["emotions"] = adjusted_emotions
                    results["Face++"]["head_pose"] = head_pose
                    results["Face++"]["eye_status"] = eye_status
                    results["Face++"]["dominant_emotion"] = dominant_emotion
                    results["Face++"]["most_prevalent_emotion"] = most_prevalent_emotion

            except Exception as e:
                logging.error(f"Error processing Face++ API request: {str(e)}")
                results["Face++"]["error"] = f"Error processing Face++ API request: {str(e)}"

            await websocket.send_json(results)
        except WebSocketDisconnect:
            logging.info("WebSocket connection closed")
            break

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
