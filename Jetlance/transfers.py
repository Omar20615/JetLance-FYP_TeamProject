import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from os.path import exists
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
FRAME_RATE = 30

def extract_coodinates(video_path):
    file_exists = exists(video_path)
    _, extension = os.path.splitext(video_path)
    if not file_exists:
        return False
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    positions = []
    frame_number = 0
    frame_interval = int(fps / 4)
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_number += 1
                minutes, seconds = divmod(timestamp, 60)
                hours, minutes = divmod(minutes, 60)
                timestamp_formatted = "{:02.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, seconds)

            # Make detection
            if extension == '.mkv':
                image = frame[1080:, :1920, :]
            else:
                image = frame

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            landmarks = results.pose_landmarks
            if landmarks:
                left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
                left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
                right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
                left_foot_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                # Store all the landmarks in a list
                l = [
                    timestamp_formatted,
                    left_wrist.x, left_wrist.y, left_wrist.z,
                    right_wrist.x, right_wrist.y, right_wrist.z,
                    left_shoulder.x, left_shoulder.y, left_shoulder.z,
                    right_shoulder.x, right_shoulder.y, right_shoulder.z,
                    left_elbow.x, left_elbow.y, left_elbow.z,
                    right_elbow.x, right_elbow.y, right_elbow.z,
                    left_hip.x, left_hip.y, left_hip.z,
                    right_hip.x, right_hip.y, right_hip.z,
                    left_knee.x, left_knee.y, left_knee.z,
                    right_knee.x, right_knee.y, right_knee.z,
                    left_ankle.x, left_ankle.y, left_ankle.z,
                    right_ankle.x, right_ankle.y, right_ankle.z,
                    left_ear.x, left_ear.y, left_ear.z,
                    right_ear.x, right_ear.y, right_ear.z,
                    left_heel.x, left_heel.y, left_heel.z,
                    right_heel.x, right_heel.y, right_heel.z,
                    left_foot_index.x, left_foot_index.y, left_foot_index.z,
                    right_foot_index.x, right_foot_index.y, right_foot_index.z
                ]
                positions.append(l)
            # cv2.imshow('Mediapipe Feed', image)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

    cap.release()
    data = pd.DataFrame(positions)
    data.columns = ["TIME_STAMP",
                    "LEFT_WRIST.x", "LEFT_WRIST.y", "LEFT_WRIST.z",
                    "RIGHT_WRIST.x", "RIGHT_WRIST.y", "RIGHT_WRIST.z",
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y","LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x","RIGHT_SHOULDER.y","RIGHT_SHOULDER.z",
                    "LEFT_ELBOW.x","LEFT_ELBOW.y","LEFT_ELBOW.z",
                    "RIGHT_ELBOW.x","RIGHT_ELBOW.y","RIGHT_ELBOW.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z",
                    "LEFT_KNEE.x","LEFT_KNEE.y","LEFT_KNEE.z",
                    "RIGHT_KNEE.x","RIGHT_KNEE.y","RIGHT_KNEE.z",
                    "LEFT_ANKLE.x","LEFT_ANKLE.y","LEFT_ANKLE.z",
                    "RIGHT_ANKLE.x","RIGHT_ANKLE.y","RIGHT_ANKLE.z",
                    "LEFT_EAR.x","LEFT_EAR.y","LEFT_EAR.z",
                    "RIGHT_EAR.x","RIGHT_EAR.y","RIGHT_EAR.z",
                    "LEFT_HEEL.x", "LEFT_HEEL.y", "LEFT_HEEL.z",
                    "RIGHT_HEEL.x","RIGHT_HEEL.y","RIGHT_HEEL.z",
                    "LEFT_FOOT_INDEX.x","LEFT_FOOT_INDEX.y","LEFT_FOOT_INDEX.z",
                    "RIGHT_FOOT_INDEX.x", "RIGHT_FOOT_INDEX.y", "RIGHT_FOOT_INDEX.z"]
    return data


# Function to calculate trajectory analysis score
def calculate_trajectory_analysis_score(df):
    # Convert DataFrame to numpy array and remove timestamp column if present
    df_no_timestamp = df.iloc[:, 1:].to_numpy()

    # Example: calculate score based on trajectory analysis
    # Calculate trajectory deviation or smoothness (e.g., Euclidean distance between consecutive joint positions)
    trajectory_deviation = np.linalg.norm(np.diff(df_no_timestamp, axis=0), axis=1)

    # Calculate average trajectory deviation
    avg_deviation = np.mean(trajectory_deviation)

    # Score based on trajectory analysis (lower is better)
    trajectory_analysis_score = avg_deviation

    return trajectory_analysis_score

def grade(data):
    tas = calculate_trajectory_analysis_score(data)
    if tas <= 0.1:
        score = 4
    elif tas <= 0.15:
        score = 3
    elif tas <= 0.2:
        score = 2
    elif tas <= 0.25:
        score = 1
    else:
        score = 0
    return score

def calculate(video_path):
    data = extract_coodinates(video_path)
    score = grade(data)
    return score

# print(calculate("E:\FYP\OBS\Subject#1\Transfers.mkv"))
