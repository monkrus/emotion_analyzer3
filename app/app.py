from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os
import logging
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_API_ENDPOINT = os.getenv("FACEPP_API_ENDPOINT")

# Ensure that the API key, secret, and endpoint are set
if not FACEPP_API_KEY or not FACEPP_API_SECRET or not FACEPP_API_ENDPOINT:
    raise ValueError("One or more environment variables for Face++ API are not set.")

# Try to load ElenaRyumina/face_emotion_recognition model
face_emotion_model = None
try:
    face_emotion_model = AutoModelForImageClassification.from_pretrained("ElenaRyumina/face_emotion_recognition")
    logging.info("Successfully loaded ElenaRyumina/face_emotion_recognition model")
except Exception as e:
    logging.error(f"Failed to load ElenaRyumina/face_emotion_recognition model: {str(e)}")

# Define image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def extract_emotion_probs(emotion_dict):
    # Calculate the dominant emotion by finding the one with the highest probability
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    return dominant_emotion

@app.post("/detect_attributes")
async def detect_attributes(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Face++ API request
        files = {'image_file': contents}
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'return_attributes': 'headpose,eyestatus,emotion'
        }
    
        response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
        response.raise_for_status()

        faces = response.json().get('faces', [])
        facepp_dominant_emotion = None
        facepp_emotion = None

        if faces:
            face_attributes = faces[0]['attributes']
            facepp_emotion = face_attributes['emotion']
            facepp_dominant_emotion = extract_emotion_probs(facepp_emotion)

        # Process image for ElenaRyumina/face_emotion_recognition Emotion Detection if model is available
        face_emotion_emotions = None
        face_emotion_dominant_emotion = None
        if face_emotion_model:
            image = Image.open(file.file).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

            face_emotion_output = face_emotion_model(image_tensor)
            face_emotion_probs = torch.nn.functional.softmax(face_emotion_output.logits, dim=1)
            face_emotion_emotions = {label: prob.item() for label, prob in zip(face_emotion_model.config.id2label.values(), face_emotion_probs[0])}
            face_emotion_dominant_emotion = extract_emotion_probs(face_emotion_emotions)

        # Compare and select the most probable emotion
        if face_emotion_emotions and facepp_emotion:
            combined_emotions = {**face_emotion_emotions, **facepp_emotion}
            dominant_emotion = extract_emotion_probs(combined_emotions)
        elif facepp_emotion:
            dominant_emotion = facepp_dominant_emotion
        elif face_emotion_emotions:
            dominant_emotion = face_emotion_dominant_emotion
        else:
            dominant_emotion = None

        return JSONResponse({
            "faceppEmotion": facepp_emotion,
            "faceEmotionRecognitionEmotion": face_emotion_emotions,
            "dominantEmotion": dominant_emotion
        })
    except requests.RequestException as e:
        logging.error(f"Request error: {str(e)}")
        return JSONResponse({"error": f"Request error: {str(e)}"}, status_code=400)
    except Exception as e:
        logging.error(f"Internal Server Error: {str(e)}")
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)