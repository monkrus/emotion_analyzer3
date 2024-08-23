from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import os
import logging
import base64

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

@app.get("/", response_class=HTMLResponse)
async def main_page():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading main page: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading page</h1><p>{str(e)}</p>", status_code=500)

def extract_dominant_attribute(attribute_dict):
    return max(attribute_dict, key=attribute_dict.get)

def get_recommendation(emotion, head_pose, eye_status):
    # ... (keep the existing get_recommendation function as is) ...
    pass  # Add this line or replace with your existing code

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
                    "recommendation": None,
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
                    results["Face++"]["emotions"] = face_attributes['emotion']
                    results["Face++"]["head_pose"] = face_attributes['headpose']
                    results["Face++"]["eye_status"] = face_attributes['eyestatus']
                    results["Face++"]["dominant_emotion"] = extract_dominant_attribute(face_attributes['emotion'])
                    results["Face++"]["recommendation"] = get_recommendation(
                        results["Face++"]["dominant_emotion"],
                        face_attributes['headpose'],
                        face_attributes['eyestatus']
                    )
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
