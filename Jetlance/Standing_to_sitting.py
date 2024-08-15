'''Sitting to Standing'''
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
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_thumb = landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB]
        right_thumb = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_wrist.x, left_wrist.y, left_wrist.z,
            right_wrist.x, right_wrist.y, right_wrist.z,
            left_thumb.x, left_thumb.y, left_thumb.z,
            right_thumb.x, right_thumb.y, right_thumb.z,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z,
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
                    "LEFT_WRIST.x", "LEFT_WRIST.y", "LEFT_WRIST.z",
                    "RIGHT_WRIST.x", "RIGHT_WRIST.y", "RIGHT_WRIST.z",
                    "LEFT_THUMB.x", "LEFT_THUMB.y","LEFT_THUMB.z",
                    "RIGHT_THUMB.x","RIGHT_THUMB.y","RIGHT_THUMB.z",
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y","LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x","RIGHT_SHOULDER.y","RIGHT_SHOULDER.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z"]

    data1.columns = column_names
    return data1


def calculate_duration(time_parts):
    time_parts = time_parts.split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def grade(df):
    initial = (df.at[1, 'LEFT_SHOULDER.x'] + df.at[1, 'LEFT_SHOULDER.x']) / 2
    prev = None
    time = None
    r = None

    for index, row in df.iterrows():
        # Get wrist coordinates for the current row
        left_shoulder_coordinates = row['LEFT_SHOULDER.x']
        right_shoulder_coordinates = row['RIGHT_SHOULDER.x']
        current = (left_shoulder_coordinates + right_shoulder_coordinates) / 2
        if current - initial >= 0.05 and abs(round(prev, 2) - round(current, 2)) == 0:
            time = row['TIME_STAMP']
            break
        prev = current
        r = index
    total_seconds = calculate_duration(time)
    std_dev_hand = (np.std(df['LEFT_WRIST.x'].loc[1:r]) + np.std(df['RIGHT_WRIST.x'].loc[1:r]))/2
    std_dev_hip = (np.std(df['LEFT_HIP.x'].loc[1:r]) + np.std(df['RIGHT_HIP.x'].loc[1:r]))/2
    if std_dev_hand < std_dev_hip:
        hand_used = True
    else:
        hand_used = False
    if total_seconds < 10 and not hand_used:
        score = 4
    elif total_seconds < 10 and hand_used:
        score = 3
    elif total_seconds < 15:
        score = 2
    elif total_seconds < 20:
        score = 1
    else:
        score = 0
    return score


def calculate(videopath):
    df = process_video(videopath)
    score = grade(df)
    return score

# video_path = "E:\FYP\OBS\Subject#1\Standing to Sitting.mkv"
# print('Score:', calculate(video_path))
