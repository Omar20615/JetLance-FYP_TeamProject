import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from os.path import exists
mp_pose = mp.solutions.pose

# Define thresholds and weights
DISTANCE_THRESHOLDS = [0, 0.2, 0.3, 0.4]
WEIGHTS = {'distance': 1.0}

def extract_joint_positions(frame, pose, timestamp_formatted):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    landmarks = results.pose_landmarks
    l = []
    if landmarks:
        left_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        right_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_index.x, left_index.y, left_index.z,
            right_index.x, right_index.y, right_index.z,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z
            ]
    return l
def process_video(video_path):
    file_exists = exists(video_path)
    _, extension = os.path.splitext(video_path)
    if not file_exists:
        return False
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    positions1 = []
    frame_number = 0
    frame_interval = int(fps / 1)
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
                if extension == '.mkv':
                    joint_positions = extract_joint_positions(frame[1080:, :1920, :], pose, timestamp_formatted)
                else:
                    joint_positions = extract_joint_positions(frame, pose, timestamp_formatted)
                positions1.append(joint_positions)
            frame_count += 1
    cap.release()
    data1 = pd.DataFrame(positions1)
    column_names = ["TIME_STAMP",
                    "LEFT_INDEX.x", "LEFT_INDEX.y", "LEFT_INDEX.z",
                    "RIGHT_INDEX.x", "RIGHT_INDEX.y", "RIGHT_INDEX.z",
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y","LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x","RIGHT_SHOULDER.y","RIGHT_SHOULDER.z"]

    data1.columns = column_names
    return data1

def normalize_score(value, thresholds):
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= value < thresholds[i+1]:
            return i + 1
    return len(thresholds)

# Calculate maximum forward distance from joint coordinates
def calculate_max_forward_distance(joint_coordinates):
    max_forward_distance = joint_coordinates['RIGHT_INDEX.y'].max() - joint_coordinates['RIGHT_INDEX.y'].min()
    columns_of_interest = ["LEFT_SHOULDER.x", "RIGHT_SHOULDER.x"]
    std_devs = joint_coordinates[columns_of_interest].std()
    return max_forward_distance, std_devs

# Function to calculate reaching forward score
def calculate_reaching_forward_score(max_forward_distance, std_devs):
    # Normalize distance score
    distance_score = normalize_score(max_forward_distance, DISTANCE_THRESHOLDS)
    # Apply supervision flag adjustment
    if any(std > 0.10 for std in std_devs):
        total_score = distance_score - 1
    else:
        total_score = distance_score

    # Ensure score is within range [0, 4]
    total_score = np.clip(total_score, 0, 4)

    return total_score

def calculate(video_path):
    joint_coordinates = process_video(video_path)
    # Calculate maximum forward distance
    max_forward_distance, std_devs = calculate_max_forward_distance(joint_coordinates)
    # Calculate reaching forward score
    reaching_forward_score = calculate_reaching_forward_score(max_forward_distance, std_devs)
    return(reaching_forward_score)


# print(calculate("E:\FYP\OBS\Subject#1\Reaching Forward with Outstretched Arm While Standing.mkv"))
