from calendar import c
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

from module import mediapipe_detection, draw_landmarks, extract_keypoints

# 變數
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Path for exported data, numpy arrays 存放資料集的資料夾名稱
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect 手勢種類名稱
actions = np.array(['open', 'fist', 'one', 'two', 'three', 'four', 'six','eight', 'nine', 'ok', 'check', 'like', 'middel', 'yo'])

# Thirty videos worth of data 每個手勢紀錄幾組
num_sequences = 15

# Videos are going to be 30 frames in length 每組紀錄幾幀
sequence_length = 3

actions_dict = dict()
max_count = 0

for action in actions: 
    folder_path = os.path.join(DATA_PATH, action)
    try: # 嘗試建立資料夾
        os.makedirs(folder_path)
        actions_dict[action] = 0
    except: # 若資料夾已經存在
        #print("folder namded:", action," already exists")
        seq_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        actions_dict[action] = seq_count
        
    #     # 紀錄數量的最大值
    #     if seq_count > max_count:
    #         max_count = seq_count

# 印出手勢種類名稱和數量
for action, seq_num in actions_dict.items():
    print(action.ljust(8), str(seq_num).center(10))

# 倒數計時
def count_down(frame_num, image, action_text):
    # 取得手勢後的第一幀：顯示 STARTING COLLECTION ! 字樣
    if frame_num < 10:
        cv2.putText(image, 'STARTING COLLECTING ' + str.upper(action_text), (100,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第二幀：顯示 3 字樣
    elif frame_num < 20: 
        cv2.putText(image, '3', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第三幀：顯示 2 字樣
    elif frame_num < 30: 
        cv2.putText(image, '2', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
    # 取得手勢後的第四幀：顯示 1 字樣
    elif frame_num < 40: 
        cv2.putText(image, '1', (300,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
    else:
        cv2.putText(image, 'GO', (280,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
        return True, image
    
    return False, image

# 遮罩
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
    # 印出 Kinect 參數
    print(device_config)

    # Start cameras using modified configuration 初始化 Kinect Camera
    pyK4A.device_start_cameras(device_config)

    # Initialize the body tracker 初始化 Kinect Skeleton
    pyK4A.bodyTracker_start(bodyTrackingModulePath)

    # 紀錄被觸發的按鍵編號
    keycode = 0

    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands:
        
        # NEW LOOP
        # Loop through actions 依序遍歷每個手勢
        for action, seq_num in actions_dict.items():
            
            frame_num = 0 #  
            sequences = []      # 儲存一組(三幀)手部關鍵點座標
            frame_skeleton = [] # 單影格的手部關鍵點座標
            sequences_count = 0 # 已經紀錄幾組資料
            is_recoding = False # 是否在錄製狀態(False = 倒數狀態)
            
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

                    # Read and convert the image data to numpy array (Kinect)
                    color_image = pyK4A.image_convert_to_numpy(color_image_handle)

                    # 彩圖 → GPU
                    #color_image = cv2.UMat(color_image)
                    

                    # Read and convert the image data to numpy array (Kinect)
                    #depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
                    #depth_image = cv2.convertScaleAbs (depth_image, alpha=0.05)

                    # Transform the depth image to the color format (Kinect)
                    transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle, color_image_handle)

                    # 深度圖 → GPU
                    #transformed_depth_image = cv2.UMat(transformed_depth_image)
                    

                    # Make detections (MediPipe)
                    image, results = mediapipe_detection(color_image, hands)

                    # Draw landmarks (MediPipe)
                    image, no_hand = draw_landmarks(image, results, mp_hands, mp_drawing_styles, mp_drawing)
                    
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequences_count), (30,24), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    # 彩圖 → GPU
                    color_image = cv2.UMat(color_image)
                    # 深度圖 → GPU
                    transformed_depth_image = cv2.UMat(transformed_depth_image)

                    # 水平翻轉
                    color_image = cv2.flip(color_image,1)
                    transformed_depth_image = cv2.flip(transformed_depth_image,1)
                
                    if is_recoding:
                        
                        if sequences_count >= num_sequences:
                        # 此類資料收集完成

                            break
                        
                        elif frame_num > (5 + sequence_length):
                        # 已經收集了一組(三幀)手部關鍵點資料
                            
                            sec = str(int(time.time()*10)) # 建立時間戳記
                                            
                            # 儲存
                            npy_path = os.path.join(DATA_PATH, action, sec) # 生成路徑                       
                            np.save(npy_path, sequences.reshape(sequence_length, 63)) # 存檔
                            
                            sequences_count += 1
                            
                            # 清空
                            frame_num  = 0
                            sequences = []
                        
                        elif frame_num > 5:
                        # 間隔五秒

                            if no_hand: 
                            # 沒有偵測到手部
                                cv2.putText(image, 'HAND NOT FOUND', (120,200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
                                frame_num  = 0
                                sequences = []
                            
                            else:
                            # 偵測到至少一個手部

                                # 提取手部關鍵點座標
                                keypoints = extract_keypoints(results)
                                # 加入暫存 List
                                sequences = np.append(sequences, keypoints)

                                # 遍歷每一個 Kinect Skeleton 找到的 Body
                                for body in pyK4A.body_tracker.bodiesNow: 

                                    sec = str(int(time.time()*100)) # 建立時間搓記

                                    # 清空
                                    frame_skeleton = []

                                    # 骨架3D點投影到 RGB Image
                                    skeleton2D_rgb = pyK4A.bodyTracker_project_skeleton_rgb(body.skeleton)
                                    # 骨架3D點投影到 Depth Image
                                    #skeleton2D_depth = pyK4A.bodyTracker_project_skeleton(body.skeleton)

                                    # 基於 MediaPipe 辨識結果判斷左右手 → 取得 Kinect Skeleton 對應的編號
                                    if results.multi_handedness[0].classification[0].label == "Right":
                                        hand_side = 15 # 右手
                                    else:
                                        hand_side = 8  # 左手

                                    # 取得手部座標
                                    joint = skeleton2D_rgb.joints2D[hand_side]
                                    joint_x = int(-joint.position.v[0]+640*2) # 垂直翻轉
                                    joint_y = int(joint.position.v[1])

                                    if joint_x-112>0 and joint_x+112<1280 and joint_y-112>0 and joint_y+112<720:

                                        #color_image = cv2.circle(color_image, (int(joint.position.v[0]), int(joint.position.v[1])), 3, (255,0,0), 3)

                                        # 裁切彩圖
                                        #crop_color_image = color_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]
                                        # in GPU
                                        crop_color_image = cv2.UMat(color_image, [joint_y-112, joint_y+112], [joint_x-112, joint_x+112])

                                        #crop_color_image_show = crop_color_image.copy()
                                        #print("color_image shape", color_image.shape)

                                        for res in results.multi_hand_landmarks[0].landmark:
                                            # 座標轉換：彩圖 → 裁切彩圖
                                            frame_skeleton.append([int(res.x*1280)-joint_x+112, int(res.y*720)-joint_y+112, res.z])
                                            
                                            # 確認座標轉換是否能對應到裁切後的彩圖
                                            #cv2.circle(crop_color_image_show, (int(res.x*1280)-joint_x+112, int(res.y*720)-joint_y+112), 2, (255, 0, 0), -1)
                                        
                                        #print(frame_skeleton)
                                        
                                        # 裁切深度圖
                                        #crop_transformed_depth_image = transformed_depth_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]
                                        crop_transformed_depth_image = cv2.UMat(transformed_depth_image, [joint_y-112, joint_y+112], [joint_x-112, joint_x+112])
                                        # 鮮明化
                                        transformed_depth_image = cv2.convertScaleAbs (transformed_depth_image, alpha=0.05)
                                        # 手部遮罩
                                        crop_transformed_depth_image = copyto(crop_transformed_depth_image, cv2.UMat.get(crop_transformed_depth_image)[112, 112])

                                        # 儲存座標
                                        np.save(os.path.join(DATA_PATH, action, sec+'_crd'), frame_skeleton)

                                        # 顯示裁切彩圖
                                        cv2.imshow("Crop Color Image", crop_color_image)
                                        #cv2.imshow("Crop Depth Image", crop_transformed_depth_image)

                                        # 儲存圖片
                                        cv2.imwrite( os.path.join(DATA_PATH, action, sec+'_rgb.jpg'), crop_color_image)
                                        cv2.imwrite( os.path.join(DATA_PATH, action, sec+'_dpt.jpg'), crop_transformed_depth_image)
                                
                    elif no_hand:
                    # 當前畫面中沒有手部

                        frame_num = 0 # 重新計時
                        
                    else:
                        is_recoding, image = count_down(frame_num, image, action) # 倒數
                        if is_recoding:
                            frame_num = 0
                
                    cv2.imshow('OpenCV Feed', image)
                    frame_num += 1

                    keycode = cv2.waitKey(10)

                    # Release the image
                    pyK4A.image_release(color_image_handle)
                    pyK4A.image_release(depth_image_handle)
                    pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)

                pyK4A.capture_release()
                pyK4A.body_tracker.release_frame()
                    
                if keycode==27 or keycode==ord('s'):  # s 跳過當前手勢
                    break
            
            if keycode==27: # Esc 結束程式
                break
        
        cv2.destroyAllWindows()
        pyK4A.device_stop_cameras()
        pyK4A.device_close()