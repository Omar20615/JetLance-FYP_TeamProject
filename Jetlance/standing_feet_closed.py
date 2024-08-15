import pickle
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
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        left_foot_index = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                # Store all the landmarks in a list
        l = [timestamp_formatted,
            left_wrist.x, left_wrist.y, left_wrist.z,
            right_wrist.x, right_wrist.y, right_wrist.z,
            left_shoulder.x, left_shoulder.y, left_shoulder.z,
            right_shoulder.x, right_shoulder.y, right_shoulder.z,
            left_elbow.x, left_elbow.y, left_elbow.z,
            right_elbow.x, right_elbow.y, right_elbow.z,
            left_hip.x, left_hip.y, left_hip.z,
            right_hip.x, right_hip.y, right_hip.z,
            left_knee.x, left_knee.y, left_knee.z,
            right_knee.x, right_knee.y, right_knee.z,
            left_ankle.x, left_ankle.y, left_ankle.z,
            right_ankle.x, right_ankle.y, right_ankle.z,
            left_ear.x, left_ear.y, left_ear.z,
            right_ear.x, right_ear.y, right_ear.z,
            left_heel.x, left_heel.y, left_heel.z,
            right_heel.x, right_heel.y, right_heel.z,
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
    column_names =  ["TIME_STAMP",
                    "LEFT_WRIST.x", "LEFT_WRIST.y", "LEFT_WRIST.z",
                    "RIGHT_WRIST.x", "RIGHT_WRIST.y", "RIGHT_WRIST.z",
                    "LEFT_SHOULDER.x", "LEFT_SHOULDER.y","LEFT_SHOULDER.z",
                    "RIGHT_SHOULDER.x","RIGHT_SHOULDER.y","RIGHT_SHOULDER.z",
                    "LEFT_ELBOW.x","LEFT_ELBOW.y","LEFT_ELBOW.z",
                    "RIGHT_ELBOW.x","RIGHT_ELBOW.y","RIGHT_ELBOW.z",
                    "LEFT_HIP.x","LEFT_HIP.y","LEFT_HIP.z",
                    "RIGHT_HIP.x","RIGHT_HIP.y","RIGHT_HIP.z",
                    "LEFT_KNEE.x","LEFT_KNEE.y","LEFT_KNEE.z",
                    "RIGHT_KNEE.x","RIGHT_KNEE.y","RIGHT_KNEE.z",
                    "LEFT_ANKLE.x","LEFT_ANKLE.y","LEFT_ANKLE.z",
                    "RIGHT_ANKLE.x","RIGHT_ANKLE.y","RIGHT_ANKLE.z",
                    "LEFT_EAR.x","LEFT_EAR.y","LEFT_EAR.z",
                    "RIGHT_EAR.x","RIGHT_EAR.y","RIGHT_EAR.z",
                    "LEFT_HEEL.x", "LEFT_HEEL.y", "LEFT_HEEL.z",
                    "RIGHT_HEEL.x","RIGHT_HEEL.y","RIGHT_HEEL.z",
                    "LEFT_FOOT_INDEX.x","LEFT_FOOT_INDEX.y","LEFT_FOOT_INDEX.z",
                    "RIGHT_FOOT_INDEX.x", "RIGHT_FOOT_INDEX.y", "RIGHT_FOOT_INDEX.z"]
    
    data1.columns = column_names
    return data1


def calculate_foot_distance(data, initial_frames):
  
    left_foot_x = data['LEFT_ANKLE.x'].values
    left_foot_y = data['LEFT_ANKLE.y'].values
    right_foot_x = data['RIGHT_ANKLE.x'].values
    right_foot_y = data['RIGHT_ANKLE.y'].values
    
    foot_distances = np.sqrt((right_foot_x - left_foot_x)**2 + (right_foot_y - left_foot_y)**2)
    initial_average_distance = np.mean(foot_distances[:initial_frames])
    deviations = np.abs(foot_distances - initial_average_distance)
    window_size = 5
    smoothed_deviations = np.convolve(deviations, np.ones(window_size)/window_size, mode='valid')
    
    average_deviation = np.mean(smoothed_deviations)
    #print(average_deviation)

    return average_deviation



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


def process_csv(dataframe, initial_frames):
    
    data = dataframe

    total_distance_left_hip = calculate_total_distance(data, 'LEFT_HIP')
    total_distance_right_hip = calculate_total_distance(data, 'RIGHT_HIP')
    total_distance_left_shoulder = calculate_total_distance(data, 'LEFT_SHOULDER')
    total_distance_right_shoulder = calculate_total_distance(data, 'RIGHT_SHOULDER')

    deviation = calculate_foot_distance(data, initial_frames)
 
    X = np.array([
        total_distance_left_hip,
        total_distance_right_hip,
        total_distance_left_shoulder,
        total_distance_right_shoulder,
        deviation  
    ]).T

    return X




def calculate(video_path):
    try:
        
        filename = 'F:/Omar main/fyp/standing_feet_closed/finalized_model_feet.sav'
        df = process_video(video_path)  # Assuming process_video is a function to extract joint coordinates

        x=process_csv(df,30)
        x = np.vstack([x, [0, 0, 0, 0, 0]])
        loaded_model = pickle.load(open(filename, 'rb'))
        predictions = loaded_model.predict(x)
        #print(predictions)
        return int(predictions[0])
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0  # Return a score of 4 in case of any error


# file_name = "F:/Omar main/fyp/videos/standing_feet_together/p3.mp4"
# print("Total score:", calculate(file_name))