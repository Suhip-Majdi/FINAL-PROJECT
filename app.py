from camera import VideoCamera
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import base64
import numpy as np
import cv2

app = Flask(__name__)


camera = None
# Route for the home page
@app.route('/')
def home_page():
    return render_template("project.html")


# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    return render_template("login.html")


# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    return render_template("signup.html")

# Generator function to stream the video frames
def gen(camera):
    while True:
        data = camera.get_frame_2()
        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Route for the webcam feed, starts only when "Scan your part" button is pressed
@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None:
        camera = VideoCamera()  # Initialize the camera only when needed
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')



# Route for uploading an image and processing it with YOLOv5
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Check if the request contains a 'file' part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400


    file = request.files['file']
    print(file)

    # Check if the file has been selected and has a filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    print(file.filename)

    img = np.frombuffer(file.read(), np.uint8)      # Read the image from the uploaded file
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)   #Decodes the array into an image using OpenCV.

    if img is None:         #Checks if the decoding failed; if so, returns an error.
        return jsonify({'error': 'Invalid image file'}), 400

    obj = VideoCamera()

    # Process the image using YOLOv5 and return the result
    processed_img_bytes, detection_info = obj.process_image(img)

    # Convert the processed image bytes to base64 for sending to frontend
    processed_img_base64 = base64.b64encode(processed_img_bytes).decode('utf-8')  #Web-Friendly Format json  /Frontend Compatibility

    # print(processed_img_bytes)
    # print(detection_info['object_name'])

    # Return the base64 image and detection info
    return jsonify({
        'image': processed_img_base64,
        'detection_info': detection_info
    })



# Route to stop the webcam and YOLOv5 processing
@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera
    if camera is not None:
        camera.__del__()  # Stop the video stream and clean up resources
        camera = None
    return jsonify({'status': 'Webcam stopped'}), 200

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
