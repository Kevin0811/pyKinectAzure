import sys

from numpy.core.fromnumeric import shape
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
#from kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2

cv2.setNumThreads(10)

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

if __name__ == "__main__":

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Open device
    pyK4A.device_open()

    # Modify camera configuration
    device_config = pyK4A.config
    #device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
    print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)

    # Initialize the body tracker
    pyK4A.bodyTracker_start(bodyTrackingModulePath)

    k = 0
    while True:
        # Get capture
        pyK4A.device_get_capture()

        # Get the depth image from the capture
        depth_image_handle = pyK4A.capture_get_depth_image()
        # Get the color image from the capture
        color_image_handle = pyK4A.capture_get_color_image()

        # Check the image has been read correctly
        if depth_image_handle and color_image_handle:

            # Perform body detection
            pyK4A.bodyTracker_update()

            # Get the information of each body
            #for body in pyK4A.body_tracker.bodiesNow:
            #    pyK4A.body_tracker.printBodyPosition(body)

            # Read and convert the Depth image data to numpy array:
            depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
            # Read and convert the RGB image data to numpy array:
            color_image = pyK4A.image_convert_to_numpy(color_image_handle)

            color_image_u = cv2.UMat(color_image)

            # cv2.convertScaleAbs Depth to uint8 (255)
            # alpha is fitted by visual comparison with Azure k4aviewer results 
            depth_color_image = cv2.convertScaleAbs (depth_image, alpha=0.05)

            # GRAY to RGB
            depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB) 

            # Get body segmentation image
            body_image_handle = pyK4A.bodyTracker_get_body_segmentation()
            #body_image_color = pyK4A.bodyTracker_get_body_segmentation_color() #(576, 640, 3)

            transformed_custom_image = pyK4A.transform_depth_to_color_custom(depth_image_handle,body_image_handle, color_image_handle)
            
            #body_image_color_gray = cv2.cvtColor(transformed_custom_image, cv2.COLOR_GRAY2RGB)
            #print(shape(body_image_color_gray))

            # Overlay body segmentation on depth image
            #combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)

            # 創建 Boolean 遮罩
            mask = transformed_custom_image > 0

            mask_u = cv2.UMat((255*mask).astype('uint8'))

            #body_image_color_cv = cv2.fromarray(body_image_color)

            # reference https://stackoverflow.com/questions/42594993/gradient-mask-blending-in-opencv-python

            # 拷貝 原圖(color_image) 作為背景圖片
            #background = color_image.copy()
            # 對背景圖執行遮罩
            #background[mask] = 0

            # 將原圖進行模糊處理
            blur_image_u = cv2.GaussianBlur(color_image_u, (15,15),cv2.BORDER_DEFAULT)
            blur_image = cv2.UMat.get(blur_image_u)
            # 將遮罩圖進行模糊處理
            blur_mask_u = cv2.GaussianBlur(mask_u, (15,15),cv2.BORDER_DEFAULT)
            # 將遮罩圖轉為三通道
            alpha_u = cv2.cvtColor(blur_mask_u, cv2.COLOR_GRAY2BGR)
            alpha = cv2.UMat.get(alpha_u)/255.0
            # 將原圖(前景)與模糊圖(背景)整合
            dst_image=(color_image*(1-alpha) + blur_image*alpha).astype('uint8')

            #dst_image = copyto(color_image, mask)
            
            #cv2.imshow('Segmented Depth Image',combined_image)
            cv2.imshow('Output Image',dst_image)
            #cv2.imshow('Alpha Mask',blur_mask)
            
            k = cv2.waitKey(1)

            # Release the image
            pyK4A.image_release(depth_image_handle)
            #pyK4A.image_release(body_image_handle)
            pyK4A.image_release(color_image_handle)
            pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)

        pyK4A.capture_release()
        pyK4A.body_tracker.release_frame()

        # Esc key to stop
        if k==27:
            cv2.destroyAllWindows()
            break

    pyK4A.device_stop_cameras()
    pyK4A.device_close()
