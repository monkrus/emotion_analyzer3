# emotion_analyzer3
Emotion_analyzer3  🤖 is a gradual update of human emotion recognition  in emotion_analyzer1 and emotion_analyzer2 

`pip install fastapi uvicorn python-dotenv requests deepface tf-keras python-multipart torchvision transformers`

Python code using FastAPI that interacts with the **Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus". It compares the results from the both APIs and outputs the most probable result. 

- final.py `uvicorn app:app --reload --port 8000`    Or just type `run` in the terminal. Initital attempt.
**Added logic to improve the output, additional graph.**

- test.py  `uvicorn test:app --reload --port 8002`   Testing ground

**Summary** 
..... TBU

**Flow**
....  TBU
**Adjusting Emotions Based on Head Pose and Eye Status**
Head Pose Adjustments:
If the pitch angle is downward, happiness is reduced.
If the pitch angle is upward, surprise and happiness are boosted.
Large yaw angle reduces happiness and increases disgust.

Eye Status Adjustments:
Wide open eyes increase surprise and fear.
Closed or nearly closed eyes introduce "tiredness" and "boredom."

The pitch and yaw angles from the head pose data are used to adjust the intensity of detected emotions:
Downward Tilt (Low Pitch): Reduces positive emotions like happiness.
Upward Tilt (High Pitch): Enhances emotions like surprise and happiness.
Large Yaw (Side-to-Side Movement): Reduces positive emotions and may increase negative emotions like disgust.

Eye Status Adjustment:

The eye openness values are used to adjust the intensity of emotions:
Wide-Open Eyes: Increases the likelihood of emotions like surprise and fear.
Closed or Nearly Closed Eyes: Shifts emotions towards tiredness, boredom, or suspicion.

**Also**:
- The current emotion data is added to a deque that keeps track of the last 5 readings.
- Emotions from the history are adjusted based on head pose and eye status.
- The most frequent or highest intensity emotion is identified from the combined results.



