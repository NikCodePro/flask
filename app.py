from flask import Flask, request, jsonify, make_response
import cv2
import numpy as np

# Import the detection function
from detect_person import detect_person_in_image

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello flask server..."

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

    # Run the detection
    person_present = detect_person_in_image(image_bgr)

    # Prepare JSON response
    response_data = {
        'person_detected': person_present
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
        # If data is not a string, we assume it's a Flask response object.
        response = make_response(data, status)
    else:
        response = make_response(data, status)

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response


if __name__ == '__main__':
    # Run the Flask server
    app.run(debug=True)
