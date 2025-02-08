from flask import Flask, request, jsonify, make_response
import cv2
import numpy as np
import os
import uuid  # for generating unique filenames

# Import the detection function
from detect_person import detect_person_in_image

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello flask server.."

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    # Manually handle CORS (or use flask-cors)
    # Check for image file in POST request
    if 'image' not in request.files:
        return _cors_response(jsonify({'error': 'No image file found'}), 400)

    file = request.files['image']
    if file.filename == '':
        return _cors_response(jsonify({'error': 'Empty filename'}), 400)

    # Convert file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return _cors_response(jsonify({'error': 'Could not decode image'}), 400)

    # 1) Save snapshot to local folder
    # Create a unique filename, e.g. snapshot_<uuid>.jpg
    snapshot_filename = f"snapshot_{uuid.uuid4()}.jpg"
    snapshot_path = os.path.join("C:/Users/nikhi_xkphcsm/Desktop/flask", snapshot_filename)

    # Use OpenCV to write the image to disk
    cv2.imwrite(snapshot_path, image_bgr)
    print(f"Snapshot saved to: {snapshot_path}")

    # 2) Run the detection
    person_present = detect_person_in_image(image_bgr)

    # 3) Prepare JSON response
    response_data = {
        'person_detected': person_present,
        'saved_image_path': snapshot_filename  # if you want to return the filename
    }

    return _cors_response(jsonify(response_data), 200)

# Handle preflight OPTIONS request
@app.route('/detect_faces', methods=['OPTIONS'])
def detect_faces_options():
    return _cors_response('', 200)

def _cors_response(data, status=200):
    """
    Helper function to add CORS headers to the response.
    """
    if not isinstance(data, str):
        response = make_response(data, status)
    else:
        response = make_response(data, status)

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

if __name__ == '__main__':
    # Create snapshots folder if it doesn't exist
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')

    # Run the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
