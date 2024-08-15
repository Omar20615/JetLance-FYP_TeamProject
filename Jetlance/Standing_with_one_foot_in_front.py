import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from os.path import exists
mp_pose = mp.solutions.pose

low_stability_threshold = 0.05
medium_stability_threshold = 0.02
high_stability_threshold = 0.01
length_of_other_foot = 25
normal_stride_width = 40

def extract_joint_positions(frame, pose, timestamp_formatted):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    landmarks = results.pose_landmarks
    l = []
    if landmarks:
        left_foot_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_foot_index.x, left_foot_index.y, left_foot_index.z,
            right_foot_index.x, right_foot_index.y, right_foot_index.z,
            left_heel.x, left_heel.y, left_heel.z,
            right_heel.x, right_heel.y, right_heel.z
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
                    "LEFT_FOOT_INDEX.x", "LEFT_FOOT_INDEX.y", "LEFT_FOOT_INDEX.z",
                    "RIGHT_FOOT_INDEX.x", "RIGHT_FOOT_INDEX.y", "RIGHT_FOOT_INDEX.z",
                    "LEFT_HEEL.x", "LEFT_HEEL.y","LEFT_HEEL.z",
                    "RIGHT_HEEL.x","RIGHT_HEEL.y","RIGHT_HEEL.z"]

    data1.columns = column_names
    return data1

def calculate_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def calculate_duration(time_stamps):
    time_parts = time_stamps.iloc[-1].split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


# Generate grade based on stability, foot placement, and duration
def generate_grade(average_distance, average_duration):
    if average_distance <= 0.2 and average_duration <=30:
        score = 4
    elif average_distance <= 0.3 and average_duration <=30:
        score = 3
    elif average_distance <= 0.4 and average_duration <=30:
        score = 2
    elif average_distance <= 0.4 and average_duration <=15:
        score = 3
    else:
        score = 0
    return score


def grade(data):

    # Iterate through DataFrame to process data
    distances = []
    for index, row in data.iterrows():
        left_heel = (row['LEFT_HEEL.x'], row['LEFT_HEEL.y'], row['LEFT_HEEL.z'])
        right_heel = (row['RIGHT_HEEL.x'], row['RIGHT_HEEL.y'], row['RIGHT_HEEL.z'])
        left_index = (row['LEFT_FOOT_INDEX.x'], row['LEFT_FOOT_INDEX.y'], row['LEFT_FOOT_INDEX.z'])
        right_index = (row['RIGHT_HEEL.x'], row['RIGHT_HEEL.y'], row['RIGHT_HEEL.z'])

        distance1 = calculate_distance(*left_heel, *right_index)
        distance2 = calculate_distance(*right_heel, *left_index)
        if (distance1 < distance2):
            distances.append(distance1)
        else:
            distances.append(distance2)
    # Evaluate foot placement
    average_distance = np.mean(distances)
    average_duration = calculate_duration(data['TIME_STAMP'])
    # Output final grade
    grade = generate_grade(average_distance, average_duration)
    return grade

def calculate(video_path):
    df = process_video(video_path)
    score = grade(df)
    return score

# print(calculate("E:\FYP\OBS\Subject#1\Standing with One Foot in Front.mkv"))
