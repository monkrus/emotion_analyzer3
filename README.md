# emotion_analyzer3
Emotion_analyzer3  🤖 is a gradual update of human emotion recognition  in emotion_analyzer1 and emotion_analyzer2 

`pip install fastapi uvicorn python-dotenv requests deepface tf-keras python-multipart torchvision`

Python code using FastAPI that interacts with the **Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus". It compares the results from the both APIs and outputs the most probable result. 

- app.py   `uvicorn app:app --reload --port 8000`    Or just type `run` in the terminal. Initital attempt.
- test.py  `uvicorn test:app --reload --port 8002`   Added additional statistics (deques of most intence feeling) for emotion recognition.
- app1.py  `uvicorn final:app --reload --port 8003`  Added logic tom improve the output.

**Summary** (for final.py)
Data Capture: Webcam frames are captured and sent to the backend.
Processing: The backend uses Face++ to detect emotions, head pose, and eye status.
Adjustment: Emotions are adjusted based on head pose and eye status.
Calculation: The most intense emotion is calculated from adjusted readings.
Communication: Results are sent back to the frontend.
Display: The frontend updates the display and charts with the latest data.

**Flow**
The Face++ API responds with a JSON object containing detected attributes:
Emotions: A dictionary of detected emotions and their intensities.
Head Pose: Pitch and yaw angles describing the head orientation.
Eye Status: Openness of the left and right eyes.

**Logic**
Head Pose Adjustment:

The pitch and yaw angles from the head pose data are used to adjust the intensity of detected emotions:
Downward Tilt (Low Pitch): Reduces positive emotions like happiness.
Upward Tilt (High Pitch): Enhances emotions like surprise and happiness.
Large Yaw (Side-to-Side Movement): Reduces positive emotions and may increase negative emotions like disgust.

Eye Status Adjustment:

The eye openness values are used to adjust the intensity of emotions:
Wide-Open Eyes: Increases the likelihood of emotions like surprise and fear.
Closed or Nearly Closed Eyes: Shifts emotions towards tiredness, boredom, or suspicion.


- The current emotion data is added to a deque that keeps track of the last 5 readings.
- Emotions from the history are adjusted based on head pose and eye status.
- The most frequent or highest intensity emotion is identified from the combined results.



