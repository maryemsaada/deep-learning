from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os


app = Flask(__name__)


model_path = os.path.join('models', 'my_model.keras')
model = load_model(model_path, compile=False)


label_binarizer = LabelBinarizer()
label_binarizer.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')  

@app.route('/video_feed')
def video_feed():
    """Video feed for the live camera."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Capture frames from the webcam and make predictions."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            
            x, y, w, h = 150, 100, 200, 200
            roi = frame[y:y+h, x:x+w]

           
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray_roi, (28, 28))
            normalized_roi = resized_roi / 255.0
            reshaped_roi = normalized_roi.reshape(1, 28, 28, 1)

            prediction = model.predict(reshaped_roi)
            predicted_class = label_binarizer.inverse_transform(prediction)
            predicted_letter = predicted_class[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_letter, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
