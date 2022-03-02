import cv2
import numpy as np
import os
import time

import mediapipe as mp


import sys
sys.path.insert(1, '../pyKinectAzure/')

from pyKinectAzure import pyKinectAzure, _k4a, postProcessing

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'

from module import mediapipe_detection, draw_landmarks, extract_keypoints, extract_world_keypoints

# 變數
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Path for exported data, numpy arrays 存放資料集的資料夾名稱
DATA_PATH = os.path.join('vTouch_Gestures_Data')

# Actions that we try to detect 手勢種類名稱
actions = np.array(['open', 'fist', 'one', 'two', 'three', 'four', 'six','eight', 'nine', 'ok', 'check', 'like', 'middel', 'yo'])

# Thirty videos worth of data 每個手勢紀錄幾組
num_sequences = 30

# Videos are going to be 30 frames in length 每組紀錄幾幀
sequence_length = 3

actions_dict = dict()
max_count = 0

for action in actions: 
    folder_path = os.path.join(DATA_PATH, action)
    try: # 嘗試建立資料夾
        os.makedirs(folder_path)
        actions_dict[action] = 0
    except: # 若資料夾已經存在則計算資料個數
        seq_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        actions_dict[action] = seq_count
        
        # 紀錄數量的最大值
        if seq_count > max_count:
            max_count = seq_count

# 印出手勢種類名稱和數量
for action, seq_num in actions_dict.items():
    print(action.ljust(8), str(seq_num).center(10))

# 倒數計時
def count_down(frame_num, image, action_text):
    # 取得手勢後的第一幀：顯示 STARTING COLLECTION ! 字樣
    if frame_num < 15:
        cv2.putText(image, 'STARTING COLLECTING ' + str.upper(action_text), (100,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第二幀：顯示 3 字樣
    elif frame_num < 30: 
        cv2.putText(image, '3', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第三幀：顯示 2 字樣
    elif frame_num < 45: 
        cv2.putText(image, '2', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第四幀：顯示 1 字樣
    elif frame_num < 60: 
        cv2.putText(image, '1', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    else:
        cv2.putText(image, 'GO', (280,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        return True, image
    
    return False, image

def copyto(scr_Image, depth):

	mask = cv2.inRange(scr_Image, depth*0.9, depth*1.1) 
	dst_Image = cv2.bitwise_and(scr_Image, scr_Image, mask = mask)

	return dst_Image

if __name__ == '__main__':
    
    # 宣告視窗
    cv2.namedWindow("OpenCV Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Crop Color Image", cv2.WINDOW_NORMAL)

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Open device
    pyK4A.device_open()

    # Modify camera configuration
    device_config = pyK4A.config
    #device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    device_config.camera_fps = 2 # 1 = 15 fps
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True # 同時調用 RGB 和 Depth 時啟動，確保兩者同步
    print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)

    # Initialize the body tracker
    pyK4A.bodyTracker_start(bodyTrackingModulePath)

    keycode = 0

    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        
        # NEW LOOP
        # Loop through actions
        for action, seq_num in actions_dict.items():
            
            frame_num = 0
            sequences = []
            frame_skeleton = []
            sequences_count = seq_num
            is_recoding = False
            
            while True:
                
                # Get capture
                pyK4A.device_get_capture()

                # Get the color image from the capture
                color_image_handle = pyK4A.capture_get_color_image()
                # Get the depth image from the capture
                depth_image_handle = pyK4A.capture_get_depth_image()

                # Check the image has been read correctly
                if color_image_handle and depth_image_handle:

                    # Perform body detection
                    pyK4A.bodyTracker_update()

                    # Read and convert the image data to numpy array:
                    color_image = pyK4A.image_convert_to_numpy(color_image_handle)
                    

                    # Read and convert the image data to numpy array:
                    #depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
                    #depth_image = cv2.convertScaleAbs (depth_image, alpha=0.05)

                    # Transform the depth image to the color format
                    transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle, color_image_handle)
                    

                    # Make detections
                    image, results = mediapipe_detection(color_image, hands)

                    # Draw landmarks
                    image, no_hand = draw_landmarks(image, results, mp_hands, mp_drawing_styles, mp_drawing)
                    #print(image.shape)
                    
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequences_count), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    color_image = cv2.flip(color_image,1)
                    transformed_depth_image = cv2.flip(transformed_depth_image,1)
                
                    if is_recoding:
                        
                        if sequences_count >= (max_count + num_sequences):
                            break
                            
                        elif frame_num > (5 + sequence_length):

                            sec = str(int(time.time()*10))

                            for body in pyK4A.body_tracker.bodiesNow:

                                frame_skeleton = []

                                # 骨架3D點投影到 RGB Image
                                skeleton2D_rgb = pyK4A.bodyTracker_project_skeleton_rgb(body.skeleton)
                                # 骨架3D點投影到 Depth Image
                                #skeleton2D_depth = pyK4A.bodyTracker_project_skeleton(body.skeleton)

                                joint = skeleton2D_rgb.joints2D[15]
                                joint_x = int(-joint.position.v[0]+640*2)
                                joint_y = int(joint.position.v[1])

                                if joint_x-112>0 and joint_x+112<1280 and joint_y-112>0 and joint_y+112<720:

                                    #color_image = cv2.circle(color_image, (int(joint.position.v[0]), int(joint.position.v[1])), 3, (255,0,0), 3)
                                    crop_color_image = color_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]
                                    #print("color_image shape", color_image.shape)

                                    for res in results.multi_hand_landmarks[0].landmark:
                                        frame_skeleton.append([int(res.x*1280)-joint_x+112, int(res.y*720)-joint_y+112, res.z])

                                        #cv2.circle(crop_color_image, (int(res.x*1280)-joint_x+112, int(res.y*720)-joint_y+112), 2, (255, 0, 0), -1)
                                    
                                    #print(frame_skeleton)

                                    crop_transformed_depth_image = transformed_depth_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]
                                    transformed_depth_image = cv2.convertScaleAbs (transformed_depth_image, alpha=0.05)

                                    crop_transformed_depth_image = copyto(crop_transformed_depth_image, crop_transformed_depth_image[112, 112])

                                    cv2.imshow("Crop Color Image", crop_color_image)
                                    #cv2.imshow("Crop Depth Image", crop_transformed_depth_image)

                                    cv2.imwrite( os.path.join(DATA_PATH, action, sec+'_rgb.jpg'), crop_color_image)
                                    cv2.imwrite( os.path.join(DATA_PATH, action, sec+'_dpt.jpg'), crop_transformed_depth_image)
                                    np.save(os.path.join(DATA_PATH, action, sec+'_single'), frame_skeleton)
                                            
                            # 存檔
                            npy_path = os.path.join(DATA_PATH, action, sec)
                            #print(npy_path)
                            
                            np.save(npy_path, sequences.reshape(sequence_length, 63))
                            
                            sequences_count += 1
                            
                            # 清空
                            frame_num  = 0
                            sequences = []
                        
                        elif frame_num > 5:
                            
                            # 等待手勢
                            if no_hand: 
                                cv2.putText(image, 'HAND NOT FOUND', (120,200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                                frame_num  = 0
                                sequences = []
                            else:
                                # NEW Export keypoints
                                keypoints = extract_world_keypoints(results)
                                sequences = np.append(sequences, keypoints)
                                #print(sequences)
                                
                    elif no_hand:
                        frame_num = 0
                        
                    else:
                        # 倒數準備
                        is_recoding, image = count_down(frame_num, image, action)
                        if is_recoding:
                            frame_num = 0
                
                    cv2.imshow('OpenCV Feed', image)
                    frame_num += 1

                    keycode = cv2.waitKey(1)

                    # Release the image
                    pyK4A.image_release(color_image_handle)
                    pyK4A.image_release(depth_image_handle)

                pyK4A.capture_release()
                pyK4A.body_tracker.release_frame()
                    
            if keycode==27:    # Esc key to stop
                break
        
        cv2.destroyAllWindows()
        pyK4A.device_stop_cameras()
        pyK4A.device_close()