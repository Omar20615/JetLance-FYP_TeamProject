# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# from os.path import exists
# import os
# mp_pose = mp.solutions.pose
#
# def extract_joint_positions(frame, pose, timestamp_formatted):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = pose.process(image)
#     image.flags.writeable = True
#     landmarks = results.pose_landmarks
#     l = []
#     if landmarks:
#         left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#         right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
#
#         # Store all the landmarks in a list
#         l = [
#             timestamp_formatted,
#             left_shoulder.x, left_shoulder.y, left_shoulder.z,
#             right_shoulder.x, right_shoulder.y, right_shoulder.z,
#             left_hip.x, left_hip.y, left_hip.z,
#             right_hip.x, right_hip.y, right_hip.z
#             ]
#     return l
# def process_video(video_path):
#     file_exists = exists(video_path)
#     _, extension = os.path.splitext(video_path)
#     if not file_exists:
#         return False
#     cap = cv2.VideoCapture(video_path)
#     positions1 = []
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_number = 0
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             if not ret:
#                 break
#             timestamp = frame_number / fps
#             frame_number += 1
#             minutes, seconds = divmod(timestamp, 60)
#             hours, minutes = divmod(minutes, 60)
#             timestamp_formatted = "{:02.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, seconds)
#             if extension == '.mkv':
#                 joint_positions = extract_joint_positions(frame[1080:, :1920, :], pose, timestamp_formatted)
#             else:
#                 joint_positions = extract_joint_positions(frame, pose, timestamp_formatted)
#             positions1.append(joint_positions)
#
#     cap.release()
#     data1 = pd.DataFrame(positions1)
#     column_names = ["TIME_STAMP",
#                     "LEFT_SHOULDER.x", "LEFT_SHOULDER.y", "LEFT_SHOULDER.z",
#                     "RIGHT_SHOULDER.x", "RIGHT_SHOULDER.y", "RIGHT_SHOULDER.z",
#                     "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
#                     "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z"]
#
#     data1.columns = column_names
#     return data1
#
# def calculate_angle(initial_theta, left_shoulder, right_shoulder, left_hip, right_hip):
#     # Assuming vectors are formed from shoulder to hip
#     vec1 = right_shoulder - left_shoulder
#     vec2 = right_hip - left_hip
#
#     # Calculate angle between vectors
#     dot_product = np.dot(vec1, vec2)
#     norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#     cos_theta = dot_product / norm_product
#     theta = np.arccos(cos_theta)
#
#     # Convert angle to degrees
#     theta_degrees = np.degrees(theta)
#
#     return theta_degrees
#
#
#
# def grade(data_frame):
#     # Set threshold for detecting significant angle change
#     threshold = 10
#
#     # Assume 'theta' is the angle of rotation, which is continuously updated during rotation
#     theta = 0  # Placeholder for actual angle calculation
#     initial_theta = theta  # Store initial angle
#
#     # Placeholder for tracking start and end times of rotation
#     start_time_clockwise = None
#     end_time_clockwise = None
#     start_time_anticlockwise = None
#     end_time_anticlockwise = None
#
#     # Placeholder for tracking rotation direction
#     clockwise = None
#
#     # Placeholder for grading the rotation
#     grade = None
#
#     # Placeholder for previous frame angle
#     prev_theta = initial_theta
#
#     # Main loop to process joint coordinates and detect rotation
#     for index, row in data_frame.iterrows():
#         # Extract joint coordinates from row
#         left_shoulder = np.array([row['LEFT_SHOULDER.x'], row['LEFT_SHOULDER.y'], row['LEFT_SHOULDER.z']])
#         right_shoulder = np.array([row['RIGHT_SHOULDER.x'], row['RIGHT_SHOULDER.y'], row['RIGHT_SHOULDER.z']])
#         left_hip = np.array([row['LEFT_HIP.x'], row['LEFT_HIP.y'], row['LEFT_HIP.z']])
#         right_hip = np.array([row['RIGHT_HIP.x'], row['RIGHT_HIP.y'], row['RIGHT_HIP.z']])
#
#         # Calculate current angle
#         current_theta = calculate_angle(initial_theta, left_shoulder, right_shoulder, left_hip, right_hip)
#         # Calculate change in angle
#         angle_change = current_theta - prev_theta
#         # Check if rotation has started
#         if abs(angle_change) > threshold:
#             if start_time_clockwise is None:
#                 parts = row['TIME_STAMP'].split(":")
#                 hours = int(parts[0])
#                 minutes = int(parts[1])
#                 seconds = float(parts[2])
#                 start_time_clockwise = hours * 3600 + minutes * 60 + seconds
#                 clockwise = True
#             elif start_time_anticlockwise is None:
#                 parts = row['TIME_STAMP'].split(":")
#                 hours = int(parts[0])
#                 minutes = int(parts[1])
#                 seconds = float(parts[2])
#                 start_time_anticlockwise = hours * 3600 + minutes * 60 + seconds
#                 clockwise = False
#
#         # Check if rotation direction has changed
#         if clockwise:
#             if angle_change < 0:
#                 parts = row['TIME_STAMP'].split(":")
#                 hours = int(parts[0])
#                 minutes = int(parts[1])
#                 seconds = float(parts[2])
#                 end_time_clockwise = hours * 3600 + minutes * 60 + seconds
#         else:
#             if angle_change > 0:
#                 parts = row['TIME_STAMP'].split(":")
#                 hours = int(parts[0])
#                 minutes = int(parts[1])
#                 seconds = float(parts[2])
#                 end_time_anticlockwise = hours * 3600 + minutes * 60 + seconds
#
#         # Update previous frame angle
#         prev_theta = current_theta
#
#     # Calculate rotation times
#     if start_time_clockwise is not None and end_time_clockwise is not None:
#         rotation_time_clockwise = end_time_clockwise - start_time_clockwise
#     if start_time_anticlockwise is not None and end_time_anticlockwise is not None:
#         rotation_time_anticlockwise = end_time_anticlockwise - start_time_anticlockwise
#
#     # Grade rotation based on times
#     if rotation_time_clockwise <= 4 and rotation_time_anticlockwise <= 4:
#         grade = 4
#     elif rotation_time_clockwise <= 4 or rotation_time_anticlockwise <= 4:
#         grade = 3
#     elif rotation_time_clockwise <= 12 or rotation_time_anticlockwise <= 12:
#         grade = 2
#     else:
#         grade = 0
#
#     return grade
#
# def calculate(video_path):
#     df = process_video(video_path)
#     score = grade(df)
#     return score
#
# print(calculate("E:\FYP\OBS\Subject#1\Turning 360 Degrees.mkv"))


import cv2
import mediapipe as mp
import numpy as np
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
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Store all the landmarks in a list
        l = [
            timestamp_formatted,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z,
            left_hip.x, left_hip.y, left_hip.z,
            right_hip.x, right_hip.y, right_hip.z
            ]
    return l
def process_video(video_path):
    file_exists = exists(video_path)
    if not file_exists:
        return False
    cap = cv2.VideoCapture(video_path)
    positions1 = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if not ret:
                break
            timestamp = frame_number / fps
            frame_number += 1
            minutes, seconds = divmod(timestamp, 60)
            hours, minutes = divmod(minutes, 60)
            timestamp_formatted = "{:02.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, seconds)
            right_positions = extract_joint_positions(frame[1080:, :1920, :], pose, timestamp_formatted)
            positions1.append(right_positions)

    cap.release()
    data1 = pd.DataFrame(positions1)
    column_names = ["TIME_STAMP",
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y", "LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x", "RIGHT_SHOULDER.y", "RIGHT_SHOULDER.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z"]

    data1.columns = column_names
    return data1

def calculate_angle(initial_theta, left_shoulder, right_shoulder, left_hip, right_hip):
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



def grade(data_frame):
    # Set threshold for detecting significant angle change
    threshold = 10

    # Assume 'theta' is the angle of rotation, which is continuously updated during rotation
    theta = 0  # Placeholder for actual angle calculation
    initial_theta = theta  # Store initial angle

    # Placeholder for tracking start and end times of rotation
    start_time_clockwise = None
    end_time_clockwise = None
    start_time_anticlockwise = None
    end_time_anticlockwise = None

    # Placeholder for tracking rotation direction
    clockwise = None

    # Placeholder for grading the rotation
    grade = None

    # Placeholder for previous frame angle
    prev_theta = initial_theta

    # Main loop to process joint coordinates and detect rotation
    for index, row in data_frame.iterrows():
        # Extract joint coordinates from row
        left_shoulder = np.array([row['LEFT_SHOULDER.x'], row['LEFT_SHOULDER.y'], row['LEFT_SHOULDER.z']])
        right_shoulder = np.array([row['RIGHT_SHOULDER.x'], row['RIGHT_SHOULDER.y'], row['RIGHT_SHOULDER.z']])
        left_hip = np.array([row['LEFT_HIP.x'], row['LEFT_HIP.y'], row['LEFT_HIP.z']])
        right_hip = np.array([row['RIGHT_HIP.x'], row['RIGHT_HIP.y'], row['RIGHT_HIP.z']])

        # Calculate current angle
        current_theta = calculate_angle(initial_theta, left_shoulder, right_shoulder, left_hip, right_hip)
        # Calculate change in angle
        angle_change = current_theta - prev_theta
        # Check if rotation has started
        if abs(angle_change) > threshold:
            if start_time_clockwise is None:
                parts = row['TIME_STAMP'].split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                start_time_clockwise = hours * 3600 + minutes * 60 + seconds
                clockwise = True
            elif start_time_anticlockwise is None:
                parts = row['TIME_STAMP'].split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                start_time_anticlockwise = hours * 3600 + minutes * 60 + seconds
                clockwise = False

        # Check if rotation direction has changed
        if clockwise:
            if angle_change < 0:
                parts = row['TIME_STAMP'].split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                end_time_clockwise = hours * 3600 + minutes * 60 + seconds
        else:
            if angle_change > 0:
                parts = row['TIME_STAMP'].split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                end_time_anticlockwise = hours * 3600 + minutes * 60 + seconds

        # Update previous frame angle
        prev_theta = current_theta

    # Calculate rotation times
    try:
        rotation_time_clockwise = end_time_clockwise - start_time_clockwise
        rotation_time_anticlockwise = end_time_anticlockwise - start_time_anticlockwise
        # Grade rotation based on times
        if rotation_time_clockwise <= 4 and rotation_time_anticlockwise <= 4:
            grade = 4
        elif rotation_time_clockwise <= 4 or rotation_time_anticlockwise <= 4:
            grade = 3
        elif rotation_time_clockwise <= 12 or rotation_time_anticlockwise <= 12:
            grade = 2
        else:
            grade = 0
    except Exception as e:
    # Code to handle other exceptions
        grade = 4

    return grade


def calculate(video_path):
    df = process_video(video_path)
    score = grade(df)
    return score


# print(calculate("E:\FYP\OBS\Subject#1\Turning 360 Degrees.mkv"))

