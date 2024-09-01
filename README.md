# emotion_analyzer3
Emotion_analyzer3  🤖 is a gradual update of human emotion recognition  in emotion_analyzer1 and emotion_analyzer2 

`pip install fastapi uvicorn python-dotenv requests deepface tf-keras python-multipart torchvision`

Python code using FastAPI that interacts with the **Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus". It compares the results from the both APIs and outputs the most probable result. 

- app.py   `uvicorn app:app --reload --port 8000`    Or just type `run` in the terminal. Initital attempt.
- test.py  `uvicorn test:app --reload --port 8002`   Added additional statistics (deques of most intence feeling) for emotion recognition.
- app1.py  `uvicorn final:app --reload --port 8003`  Added logic tom improve the output.

New Logic Integration
Head Pose:
If the head is tilted significantly downwards (low pitch), it might reduce the intensity of positive emotions like happiness, shifting them towards a more neutral or subdued state.
If the head is tilted upwards (high pitch), it might enhance emotions like surprise or happiness.
A large yaw (side-to-side movement) might indicate skepticism, potentially reducing the intensity of positive emotions or increasing negative ones like distrust.
Eye Status:
Wide-open eyes could increase the likelihood of emotions like surprise or fear, even if the detected emotion is neutral.
Closed or nearly closed eyes could shift the detected emotion towards tiredness, boredom, or suspicion.

