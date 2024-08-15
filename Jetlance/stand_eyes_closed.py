from statistics import mean
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
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
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,            
            left_ankle.x, left_ankle.y, left_ankle.z,
            right_ankle.x, right_ankle.y, right_ankle.z,
            left_hip.x, left_hip.y, left_hip.z,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_hip.x, right_hip.y, right_hip.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z
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
                    "LEFT_ANKLE.x","LEFT_ANKLE.y","LEFT_ANKLE.z",
                    "RIGHT_ANKLE.x","RIGHT_ANKLE.y","RIGHT_ANKLE.z",
                    "LEFT_HIP.x","LEFT_SHOULDER.x","LEFT_HIP.y","LEFT_SHOULDER.y",
                    "RIGHT_HIP.x","RIGHT_SHOULDER.x","RIGHT_HIP.y","RIGHT_SHOULDER.y",
                    "RIGHT_HIP.z","RIGHT_SHOULDER.z","LEFT_HIP.z","RIGHT_HIP.z",
                    ]

    data1.columns = column_names
    return data1



def calculate_total_distance(data, point):

    total_distance = 0
    previous_x = data.iloc[0][f'{point}.x']
    previous_y = data.iloc[0][f'{point}.y']

    for i in range(1, len(data)):
        current_x = data.iloc[i][f'{point}.x']
        current_y = data.iloc[i][f'{point}.y']

        distance = ((current_x - previous_x)**2 + (current_y - previous_y)**2)**0.5
        total_distance += distance

    return total_distance


def calculate_distance_score(total_distance):
    if total_distance < 3.5:
        return 4
    elif total_distance < 9:
        return 3
    elif total_distance < 15:
        return 2
    else:
        return 1


def calculate_footdistance_score(total_distance):
    if total_distance < 3:
        return 4
    elif total_distance < 6:
        return 3
    elif total_distance < 17:
        return 2
    else:
        return 1
    
def process_csv(dataframe):
    
    data = dataframe

        # Calculate total distance traveled by the left hip, right hip, left shoulder, right shoulder, left foot and right foot
    total_distance_left_hip = calculate_total_distance(data, 'LEFT_HIP')
    total_distance_right_hip = calculate_total_distance(data, 'RIGHT_HIP')
    total_distance_left_shoulder = calculate_total_distance(data, 'LEFT_SHOULDER')
    total_distance_right_shoulder = calculate_total_distance(data, 'RIGHT_SHOULDER')
    total_distance_left_foot = calculate_total_distance(data, 'LEFT_ANKLE')  # Added for left foot
    total_distance_right_foot = calculate_total_distance(data, 'RIGHT_ANKLE')  # Added for right foot

        # Calculate score based on total distance traveled for each body part
    distance_score_left_hip = calculate_distance_score(total_distance_left_hip)
    distance_score_right_hip = calculate_distance_score(total_distance_right_hip)
    distance_score_left_shoulder = calculate_distance_score(total_distance_left_shoulder)
    distance_score_right_shoulder = calculate_distance_score(total_distance_right_shoulder)
    distance_score_left_foot = calculate_footdistance_score(total_distance_left_foot) # Added for left foot
    distance_score_right_foot = calculate_footdistance_score(total_distance_right_foot)  # Added for right foot

        # Combine the distance scores
    distance_score = round(mean([distance_score_left_hip, distance_score_right_hip, distance_score_left_shoulder, distance_score_right_shoulder, distance_score_left_foot, distance_score_right_foot]))  # Added for left and right foot

    return distance_score


def calculate(video_path):
    try: 
        df = process_video(video_path)  # Assuming process_video is a function to extract joint coordinates
        score=process_csv(df)
        #print(score)
        return score
    except Exception as e:
        #print(f"An error occurred: {e}")
        return 0  # Return a score of 4 in case of any error


# file_name = "F:/Omar main/fyp/videos/standing_eyes_closed/p1.mp4"
# print("Total score:", calculate(file_name))