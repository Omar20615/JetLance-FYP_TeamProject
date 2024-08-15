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
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]


        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z,
            left_hip.x, left_hip.y, left_hip.z,
            right_hip.x, right_hip.y, right_hip.z,
            left_index.x, left_index.y, left_index.z,
            right_index.x, right_index.y, right_index.z
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
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y", "LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x", "RIGHT_SHOULDER.y", "RIGHT_SHOULDER.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z",
                    "LEFT_INDEX.x","LEFT_INDEX.y","LEFT_INDEX.z",
                    "RIGHT_INDEX.x","RIGHT_INDEX.y","RIGHT_INDEX.z"]

    data1.columns = column_names
    return data1

def calculate_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    # Assuming vectors are formed from shoulder to hip
    vec1 = right_shoulder - left_shoulder
    vec2 = right_hip - left_hip

    # Calculate angle between vectors
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = dot_product / norm_product
    theta = np.arccos(cos_theta)

    # Convert angle to degrees
    theta_degrees = np.degrees(theta)

    return theta_degrees


def assess_stability(data_frame):
    # Extract shoulder coordinates
    left_coords = data_frame[['LEFT_INDEX.x', 'LEFT_INDEX.y', 'LEFT_INDEX.z']].values
    right_coords = data_frame[['RIGHT_INDEX.x', 'RIGHT_INDEX.y', 'RIGHT_INDEX.z']].values

    # Calculate standard deviation of shoulder coordinates
    left_std = np.std(left_coords, axis=0)
    right_std = np.std(right_coords, axis=0)
    # Calculate overall stability score based on shoulder standard deviations
    stability_score = max(np.max(left_std), np.max(right_std))

    return stability_score
def grade(data_frame):
    # Set threshold for detecting significant angle change
    threshold = 5
    # Placeholder for tracking start and end times of left and right turns
    start_time_b = None
    end_time_b = None
    start_time_f = None
    end_time_f = None
    # Placeholder for previous angle
    prev_angle = None
    flag = 0
    max_angle = None
    # Main loop to process joint coordinates and detect turns
    for index, row in data_frame.iterrows():
        # Extract joint coordinates from row
        left_shoulder = np.array([row['LEFT_SHOULDER.x'], row['LEFT_SHOULDER.y'], row['LEFT_SHOULDER.z']])
        right_shoulder = np.array([row['RIGHT_SHOULDER.x'], row['RIGHT_SHOULDER.y'], row['RIGHT_SHOULDER.z']])
        left_hip = np.array([row['LEFT_HIP.x'], row['LEFT_HIP.y'], row['LEFT_HIP.z']])
        right_hip = np.array([row['RIGHT_HIP.x'], row['RIGHT_HIP.y'], row['RIGHT_HIP.z']])

        # Calculate current angle
        current_angle = calculate_angle(left_shoulder, right_shoulder, left_hip, right_hip)
        if max_angle is None or max_angle < current_angle:
            max_angle = current_angle
        # Check for significant angle change indicating a turn
        if prev_angle is not None and abs(current_angle - prev_angle) > threshold:
            # Check if the angle change is from left to right or vice versa
            if current_angle > prev_angle:
                # Assume it's a turn to the right
                if flag % 2 == 0:
                    start_time_f = None
                    start_time_b = None
                if start_time_f is None:
                    start_time_f = row['TIME_STAMP']
                    flag = flag + 1
                end_time_f = row['TIME_STAMP']
            else:
                # Assume it's a turn to the left
                if start_time_b is None:
                    start_time_b = row['TIME_STAMP']
                    flag = flag + 1
                end_time_b = row['TIME_STAMP']

        # Update previous angle for next iteration
        prev_angle = current_angle

    stability_score = assess_stability(data_frame)
    # Determine grade based on duration of left and right turns
    if flag == 4 and max_angle > 35:
        grade = 4  # Turns to both left and right
    elif flag == 2 and max_angle > 35:
        grade = 3  # Turns to one side only
    elif flag == 4 and max_angle < 35:
        grade = 2  # No significant turns detected
    elif stability_score > 0.1:
        grade = 1
    else:
        grade = 0
    return grade


def calculate(video_path):
    df = process_video(video_path)
    score = grade(df)
    return score


# print("Grade:", calculate("E:\FYP\OBS\Subject#1\Turning to Look Behind While Standing.mkv"))
