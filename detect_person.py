# detect_person.py

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_person_in_image(image_bgr):
    """
    Takes an OpenCV BGR image and returns True if a person (pose) is detected,
    otherwise False.
    """

    # Convert BGR to RGB for Mediapipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Initialize Mediapipe Pose in static mode (one image)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:  # If any pose landmarks are detected
            return True
        else:
            return False
