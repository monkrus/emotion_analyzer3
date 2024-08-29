# emotion_analyzer3
Emotion_analyzer3  🤖 is a gradual update of human emotion recognition  in emotion_analyzer1 and emotion_analyzer2 

`pip install fastapi uvicorn python-dotenv requests deepface tf-keras python-multipart torchvision`


`uvicorn app:app --reload --port 8000` or just type `run` in the terminal
`uvicorn test:app --reload --port 8002` (for test.py)

Python code using FastAPI that interacts with the **Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus". It compares the results from the both APIs and outputs the most probable result. (app.py)'