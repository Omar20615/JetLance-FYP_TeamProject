import cv2
import mediapipe as mp
import os
import pandas as pd
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
        left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            round(left_heel.x,1), round(left_heel.y,1),
            round(right_heel.x,1), round(right_heel.y,1),
            round(left_hip.x,1), round(left_hip.y,1),
            round(right_hip.x,1), round(right_hip.y,1)
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
    previous_joint_positions = []
    start_time = None
    end_time = None
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
                current_joint_positions = joint_positions[1:]
                if previous_joint_positions is None:
                    start_time = timestamp_formatted
                elif any(abs(elem1 - elem2) > 0.11 for elem1, elem2 in zip(current_joint_positions, previous_joint_positions)):
                    start_time = timestamp_formatted
                else:
                    if start_time is None:
                        start_time = timestamp_formatted
                        end_time = timestamp_formatted
                    else:
                        end_time = timestamp_formatted
                previous_joint_positions = current_joint_positions.copy()
                positions1.append(joint_positions)
            frame_count += 1
    cap.release()
    data1 = pd.DataFrame(positions1)
    column_names = ["TIME_STAMP",
                    "LEFT_HEEL.x", "LEFT_HEEL.y",
                    "RIGHT_HEEL.x","RIGHT_HEEL.y",
                    "LEFT_HIP.x","LEFT_HIP.y",
                    "RIGHT_HIP.x","RIGHT_HIP.y"]

    data1.columns = column_names
    total_seconds = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()
    if total_seconds >= 115:
        sitting_score = 4
    elif total_seconds >= 95:
        sitting_score = 3
    elif total_seconds >= 25:
        sitting_score = 2
    elif total_seconds >= 10:
        sitting_score = 1
    elif total_seconds < 10:
        sitting_score = 0
    else:
        sitting_score = 0
    return sitting_score


def calculate(video_path):
    score = process_video(video_path)
    return score

# path = "E:\FYP\OBS\Subject#1\Sitting with Back Unsupported but Feet on the Floor (1).mkv"
# print(calculate(path))

