import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from os.path import exists
mp_pose = mp.solutions.pose

def extract_joint_positions(frame, pose, timestamp_formatted):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    landmarks = results.pose_landmarks
    l = []
    if landmarks:
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_ankle.x, left_ankle.y, left_ankle.z,
            right_ankle.x, right_ankle.y, right_ankle.z,
            left_knee.x, left_knee.y, left_knee.z,
            right_knee.x, right_knee.y, right_knee.z,
            left_hip.x, left_hip.y, left_hip.z,
            right_hip.x, right_hip.y, right_hip.z
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
                if extension == '.mkv':
                    joint_positions = extract_joint_positions(frame[1080:, :1920, :], pose, timestamp_formatted)
                else:
                    joint_positions = extract_joint_positions(frame, pose, timestamp_formatted)
                positions1.append(joint_positions)
            frame_count += 1
    cap.release()
    data1 = pd.DataFrame(positions1)
    column_names = ["TIME_STAMP",
                    "LEFT_ANKLE.x", "LEFT_ANKLE.y", "LEFT_ANKLE.z",
                    "RIGHT_ANKLE.x", "RIGHT_ANKLE.y", "RIGHT_ANKLE.z",
                    "LEFT_KNEE.x", "LEFT_KNEE.y", "LEFT_KNEE.z",
                    "RIGHT_KNEE.x", "RIGHT_KNEE.y", "RIGHT_KNEE.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z"]

    data1.columns = column_names
    return data1

# Function to calculate angle between vectors
def calculate_angle(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = dot_product / norm_product
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

# Function to calculate knee angles from joint coordinates
def calculate_knee_angle(hip_coord, knee_coord, ankle_coord):
    thigh_vector = knee_coord - hip_coord
    shin_vector = ankle_coord - knee_coord
    return calculate_angle(thigh_vector, shin_vector)

# Grade performance based on step count and time taken
def generate_grade(step_count, time_taken):
    if step_count >= 8 and time_taken <= 20:
        return 4
    elif step_count >= 8:
        return 3
    elif step_count >= 4:
        return 2
    elif step_count >= 2:
        return 1
    else:
        return 0

def grade(data):
    # Initialize variables
    step_count = 0
    time_taken = 0
    last_right_knee_angle = 10
    last_left_knee_angle = 10

    # Iterate through DataFrame to process data
    for index, row in data.iterrows():
        # Calculate knee angles
        right_hip_coord = np.array([row['RIGHT_HIP.x'], row['RIGHT_HIP.y']])
        right_knee_coord = np.array([row['RIGHT_KNEE.x'], row['RIGHT_KNEE.y']])
        right_ankle_coord = np.array([row['RIGHT_ANKLE.x'], row['RIGHT_ANKLE.y']])
        right_knee_angle = calculate_knee_angle(right_hip_coord, right_knee_coord, right_ankle_coord)

        left_hip_coord = np.array([row['LEFT_HIP.x'], row['LEFT_HIP.y'], row['LEFT_HIP.z']])
        left_knee_coord = np.array([row['LEFT_KNEE.x'], row['LEFT_KNEE.y'], row['LEFT_KNEE.z']])
        left_ankle_coord = np.array([row['LEFT_ANKLE.x'], row['LEFT_ANKLE.y'], row['LEFT_ANKLE.z']])
        left_knee_angle = calculate_knee_angle(left_hip_coord, left_knee_coord, left_ankle_coord)
        # Check if right foot steps on stool
        if last_right_knee_angle <= 30 and right_knee_angle >= 60:
            step_count += 1

        # Check if left foot steps on stool
        if last_left_knee_angle <= 30 and left_knee_angle >= 60:
            step_count += 1

        last_right_knee_angle = right_knee_angle
        last_left_knee_angle = left_knee_angle

        # Calculate total time taken

        time_parts = row['TIME_STAMP'].split(":")
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds

        if step_count == 8:
            break

    # Generate and output grade
    grade = generate_grade(step_count, total_seconds)
    return grade

def calculate(video_path):
    df = process_video(video_path)
    score = grade(df)
    return score

# print(calculate("E:\FYP\OBS\Subject#1\Placing Alternate Foot on Stool While Standing.mkv"))
