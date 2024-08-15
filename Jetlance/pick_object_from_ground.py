import pickle
from statistics import mean
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore
from os.path import exists
mp_pose = mp.solutions.pose
FPS = 30


def extract_joint_positions(frame, pose, timestamp_formatted):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    landmarks = results.pose_landmarks
    l = []
    if landmarks:
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_foot_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        l = [
            timestamp_formatted,
            left_wrist.x, left_wrist.y, left_wrist.z,
            right_wrist.x, right_wrist.y, right_wrist.z,
            left_knee.x, left_knee.y, left_knee.z,
            right_knee.x, right_knee.y, right_knee.z,
            left_foot_index.x, left_foot_index.y, left_foot_index.z,
            right_foot_index.x, right_foot_index.y, right_foot_index.z
            ]
        
    return l


def process_video(video_path):
    file_exists = exists(video_path)
    _, extension = os.path.splitext(video_path)
    if not file_exists:
        return False
    cap = cv2.VideoCapture(video_path)
    positions1 = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_number / fps
            frame_number += 1
            minutes, seconds = divmod(timestamp, 60)
            hours, minutes = divmod(minutes, 60)
            timestamp_formatted = "{:02.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, seconds)
            if extension == '.mp4' or '.mkv':
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                right_positions = extract_joint_positions(rotated_frame, pose, timestamp_formatted)
            positions1.append(right_positions)

    cap.release()
    data1 = pd.DataFrame(positions1)
    
    column_names = ["TIME_STAMP",
                    "LEFT_WRIST.x", "LEFT_WRIST.y", "LEFT_WRIST.z",
                    "RIGHT_WRIST.x", "RIGHT_WRIST.y", "RIGHT_WRIST.z",
                    "LEFT_KNEE.x","LEFT_KNEE.y","LEFT_KNEE.z",
                    "RIGHT_KNEE.x","RIGHT_KNEE.y","RIGHT_KNEE.z",
                    "LEFT_FOOT_INDEX.x","LEFT_FOOT_INDEX.y","LEFT_FOOT_INDEX.z",
                    "RIGHT_FOOT_INDEX.x", "RIGHT_FOOT_INDEX.y", "RIGHT_FOOT_INDEX.z"]

    data1.columns = column_names
    return data1



def process_body_parts(data):
    column_names = ['LEFT_WRIST.y', 'RIGHT_WRIST.y']
    averages = {}
    for column_name in column_names:
        y = data[column_name].dropna().values  # Drop any NaN values
        top_5_max = np.sort(y)[-5:]
        z_scores = zscore(top_5_max)
        top_5_max_no_outliers = top_5_max[(z_scores > -2) & (z_scores < 2)]
        averages[column_name] = np.mean(top_5_max_no_outliers)
    
    feet_columns = ['LEFT_FOOT_INDEX.y', 'RIGHT_FOOT_INDEX.y']
    for column_name in feet_columns:
        y = data[column_name].dropna().values  # Drop any NaN values
        top_300_max = np.sort(y)[-300:]
        z_scores = zscore(top_300_max)
        top_300_max_no_outliers = top_300_max[(z_scores > -2) & (z_scores < 2)]
        averages[column_name] = np.mean(top_300_max_no_outliers)

    avg_feet = np.mean([averages['LEFT_FOOT_INDEX.y'], averages['RIGHT_FOOT_INDEX.y']])
    max_wrist = max(averages['LEFT_WRIST.y'], averages['RIGHT_WRIST.y'])   
    # print(avg_feet,max_wrist)
    return avg_feet ,max_wrist


def subtract_knee_from_wrist(data):
    wrist_columns = ['LEFT_WRIST.x', 'RIGHT_WRIST.x']
    knee_columns = ['LEFT_KNEE.x', 'RIGHT_KNEE.x']
    
    min_distance = float('inf')
    for wrist_x, knee_x in zip(wrist_columns, knee_columns):
        wrist_x_values = data[wrist_x].dropna().values
        knee_x_values = data[knee_x].dropna().values
        distances = np.sqrt((wrist_x_values - knee_x_values)**2)
        min_distance = min(min_distance, np.min(distances))
    
    return min_distance


def process_csv(dataframe):
    scores= 0
    diff_threshold = 0.00025
    data = dataframe

    diff = subtract_knee_from_wrist(data)
    avg_feet ,max_wrist = process_body_parts(data)
    if max_wrist>= (avg_feet-0.055):
        if diff > diff_threshold: 
            # print("score is : 4")
            scores = 4
        else:
            # print("score is : 3")
            scores =3

    elif max_wrist>= (avg_feet-0.15) :
        if diff > diff_threshold:
            # print("score is : 2")
            scores = 2

        else:
            # print("score is : 1")
            scores =1

    else:
        # print("score is : 0")
        scores = 0

    # print("\n")
    # print(scores)
    return scores



def calculate(video_path):
    try:
        df = process_video(video_path)  # Assuming process_video is a function to extract joint coordinates
        x=process_csv(df)
        return int(x)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0  # Return a score of 4 in case of any error


file_name = "F:/Omar main/fyp/videos/pick_up/3_score.mp4"
print("Total score:", calculate(file_name))