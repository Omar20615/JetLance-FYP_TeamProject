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
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_foot_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            right_ankle.x, right_ankle.y, right_ankle.z,
            left_foot_index.x, left_foot_index.y, left_foot_index.z,
            right_foot_index.x, right_foot_index.y, right_foot_index.z,
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
                    "RIGHT_ANKLE.x", "RIGHT_ANKLE.y", "RIGHT_ANKLE.z",
                    "LEFT_FOOT_INDEX.x", "LEFT_FOOT_INDEX.y", "LEFT_FOOT_INDEX.z",
                    "RIGHT_FOOT_INDEX.x", "RIGHT_FOOT_INDEX.y", "RIGHT_FOOT_INDEX.z",
                    "LEFT_HIP.x","LEFT_SHOULDER.x","LEFT_HIP.y","LEFT_SHOULDER.y",
                    "RIGHT_HIP.x","RIGHT_SHOULDER.x","RIGHT_HIP.y","RIGHT_SHOULDER.y",
                    "RIGHT_HIP.z","RIGHT_SHOULDER.z","LEFT_HIP.z","RIGHT_HIP.z",
                    ]

    data1.columns = column_names
    return data1

def calculate_distance_score(total_distance):
    if total_distance < 35:
        return 4
    elif total_distance < 60:
        return 3
    elif total_distance < 80:
        return 2
    elif total_distance < 100:
        return 1
    else: 
        return 0

def calculate_total_distance(data, point):
    initial_x = data.iloc[0][f'{point}.x']
    initial_y = data.iloc[0][f'{point}.y']
    total_distance = 0

    for i in range(1, len(data)):
        current_x = data.iloc[i][f'{point}.x']
        current_y = data.iloc[i][f'{point}.y']

        distance = ((current_x - initial_x)**2 + (current_y - initial_y)**2)**0.5
        total_distance += distance

    return total_distance

def calculate_score(pd):
    score_lift = leg_lift_score(pd)
    total_distances = {
        'left_hip': [],
        'right_hip': [],
        'left_shoulder': [],
        'right_shoulder': [],
    }

    data = pd
    for part in total_distances.keys():
        total_distances[part] = calculate_total_distance(data, part.upper().replace('_', '_'))

    distance_scores = {
        'left_hip': calculate_distance_score(round(total_distances['left_hip'])),
        'right_hip': calculate_distance_score(round(total_distances['right_hip'])),
        'left_shoulder': calculate_distance_score(round(total_distances['left_shoulder'])),
        'right_shoulder': calculate_distance_score(round(total_distances['right_shoulder'])),
    }

    score_standing = round(mean(distance_scores.values()))
    total_score = score_lift
    if total_score == 4 and score_standing == 2:
        total_score = 3
    elif total_score == 4 and score_standing == 1:
        total_score = 2
    elif total_score == 3 and score_standing == 1:
        total_score = 2
    elif score_standing == 0:
        total_score = score_lift - 2

    
    return total_score

def leg_lift_score(df):
    left_ankle_y = []
    right_ankle_y = []

    left_ankle_y = df['LEFT_FOOT_INDEX.y']
    right_ankle_y = df['RIGHT_FOOT_INDEX.y']


    left_ankle_y = np.array(left_ankle_y)
    right_ankle_y = np.array(right_ankle_y)

    half_length = len(left_ankle_y) // 2
    left_ankle_y_half1 = left_ankle_y[:half_length]
    left_ankle_y_half2 = left_ankle_y[half_length:]
    right_ankle_y_half1 = right_ankle_y[:half_length]
    right_ankle_y_half2 = right_ankle_y[half_length:]

    left_counts = []
    right_counts = []

    left_std_half1 = np.std(left_ankle_y_half1)
    left_std_half2 = np.std(left_ankle_y_half2)
    right_std_half1 = np.std(right_ankle_y_half1)
    right_std_half2 = np.std(right_ankle_y_half2)

    left_lifting = False
    left_lift_start_frame = 0
    left_lift_frame_count = 0
    left_lift_values = []

    right_lifting = False
    right_lift_start_frame = 0
    right_lift_frame_count = 0
    right_lift_values = []

    if left_std_half1 > left_std_half2:
        for frame, value in enumerate(left_ankle_y):
            if not left_lifting and np.abs(value - left_ankle_y[:1*FPS][-1]) > left_std_half1:
                left_lifting = True
                left_lift_start_frame = frame
            elif left_lifting and np.abs(value - left_ankle_y[:1*FPS][-1]) <= left_std_half1:
                left_lifting = False
                left_lift_frame_count += frame - left_lift_start_frame
                left_lift_values.extend(left_ankle_y_half1[left_lift_start_frame:frame])
                left_counts.append(left_lift_frame_count)
                left_lift_frame_count = 0

        if left_lifting:
            left_lift_frame_count += frame - left_lift_start_frame + 1
            left_lift_values.extend(left_ankle_y_half2[left_lift_start_frame:frame+1])
            left_counts.append(left_lift_frame_count)
    else:
        for frame, value in enumerate(left_ankle_y_half2):
            v = np.abs(value - left_ankle_y[:1*FPS][-1])
            if not left_lifting and v > left_std_half2:
                left_lifting = True
                left_lift_start_frame = frame
            elif left_lifting and v <= left_std_half2:
                left_lifting = False
                left_lift_frame_count += frame - left_lift_start_frame
                left_lift_values.extend(left_ankle_y_half2[left_lift_start_frame:frame])
                left_counts.append(left_lift_frame_count)
                left_lift_frame_count = 0

        if left_lifting:
            left_lift_frame_count += frame - left_lift_start_frame + 1
            left_lift_values.extend(left_ankle_y_half2[left_lift_start_frame:frame+1])
            left_counts.append(left_lift_frame_count)

    if right_std_half1 > right_std_half2:
        for frame, value in enumerate(right_ankle_y):
            if not right_lifting and np.abs(value - right_ankle_y[:1*FPS][-1]) > right_std_half1:
                right_lifting = True
                right_lift_start_frame = frame
            elif right_lifting and np.abs(value - right_ankle_y[:1*FPS][-1]) <= right_std_half1:
                right_lifting = False
                right_lift_frame_count += frame - right_lift_start_frame
                right_lift_values.extend(right_ankle_y_half1[right_lift_start_frame:frame])
                right_counts.append(right_lift_frame_count)
                right_lift_frame_count = 0

        if right_lifting:
            right_lift_frame_count += frame - right_lift_start_frame + 1
            right_lift_values.extend(right_ankle_y_half2[right_lift_start_frame:frame+1])
            right_counts.append(right_lift_frame_count)
    else:
        for frame, value in enumerate(right_ankle_y_half2):
            if not right_lifting and np.abs(value - right_ankle_y[:1*FPS][-1]) > right_std_half2:
                right_lifting = True
                right_lift_start_frame = frame
            elif right_lifting and np.abs(value - right_std_half2) <= right_std_half2:
                right_lifting = False
                right_lift_frame_count += frame - right_lift_start_frame
                right_lift_values.extend(right_ankle_y_half2[right_lift_start_frame:frame])
                right_counts.append(right_lift_frame_count)
                right_lift_frame_count = 0

        if right_lifting:
            right_lift_frame_count += frame - right_lift_start_frame + 1
            right_lift_values.extend(right_ankle_y_half2[right_lift_start_frame:frame+1])
            right_counts.append(right_lift_frame_count)

    # Calculate time duration of leg lift for each leg
    left_lift_duration_sec = max(left_counts) / FPS if left_counts else 0
    right_lift_duration_sec = max(right_counts) / FPS if right_counts else 0

    # print("Left leg lift duration:", left_lift_duration_sec, "seconds")
    # print("Right leg lift duration:", right_lift_duration_sec, "seconds")

    left_score = 0
    if left_lift_duration_sec > 10:
        left_score = 4
    elif left_lift_duration_sec > 5:
        left_score = 3
    elif left_lift_duration_sec > 3:
        left_score = 2
    elif left_lift_duration_sec > 1:
        left_score = 1
    else:
        left_score = 0

    right_score = 0
    if right_lift_duration_sec > 10:
        right_score = 4
    elif right_lift_duration_sec > 5:
        right_score = 3
    elif right_lift_duration_sec > 3:
        right_score = 2
    elif right_lift_duration_sec > 1:
        right_score = 1
    else:
        right_score = 0

    total_score = round(mean([right_score, left_score]))
    return total_score

def calculate(video_path):
    try:
        df = process_video(video_path)  # Assuming process_video is a function to extract joint coordinates
        score = calculate_score(df)  # Assuming grade is a function to assess the performance based on joint coordinates
        return int(score)
    except Exception as e:
       # print(f"An error occurred: {e}")
        return 0  # Return a score of 4 in case of any error

# file_name = "F:/Omar main/fyp/videos/1leg_m/p4.mp4"
# print("Total score:", calculate(file_name))