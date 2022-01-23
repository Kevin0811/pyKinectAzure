import sys
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a, postProcessing
import cv2

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

def copyto(scr_Image, depth):

	mask = cv2.inRange(scr_Image, depth*0.9, depth*1.1) 
	dst_Image = cv2.bitwise_and(scr_Image, scr_Image, mask = mask)

	return dst_Image

if __name__ == "__main__":

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Open device
    pyK4A.device_open()

    # Modify camera configuration
    device_config = pyK4A.config
    #device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    device_config.camera_fps = 1 # 1 = 15 fps
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True # 同時調用 RGB 和 Depth 時啟動，確保兩者同步
    print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)

    # Initialize the body tracker
    pyK4A.bodyTracker_start(bodyTrackingModulePath)

    keycode = 0
    frame_cnt = 0

    while True:

        frame_cnt += 1

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
            transformed_depth_image = cv2.convertScaleAbs (transformed_depth_image, alpha=0.05)

            # Smooth the image using Navier-Stokes based inpainintg. maximum_hole_size defines 
			# the maximum hole size to be filled, bigger hole size will take longer time to process
            #maximum_hole_size = 10
            #transformed_depth_image = postProcessing.smooth_depth_image(transformed_depth_image, maximum_hole_size)

            for body in pyK4A.body_tracker.bodiesNow:
                # 骨架3D點投影到 RGB Image
                skeleton2D_rgb = pyK4A.bodyTracker_project_skeleton_rgb(body.skeleton)
                # 骨架3D點投影到 Depth Image
                #skeleton2D_depth = pyK4A.bodyTracker_project_skeleton(body.skeleton)

                joint = skeleton2D_rgb.joints2D[15]
                joint_x = int(joint.position.v[0])
                joint_y = int(joint.position.v[1])

                if joint_x-112>0 and joint_x+112<1280 and joint_y-112>0 and joint_y+112<720:

                    #color_image = cv2.circle(color_image, (int(joint.position.v[0]), int(joint.position.v[1])), 3, (255,0,0), 3)
                    crop_color_image = color_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]
                    crop_transformed_depth_image = transformed_depth_image[joint_y-112:joint_y+112, joint_x-112:joint_x+112]

                    crop_transformed_depth_image = copyto(crop_transformed_depth_image, crop_transformed_depth_image[112, 112])

                    cv2.namedWindow('Crop Color Image',cv2.WINDOW_NORMAL)
                    cv2.imshow("Crop Color Image", crop_color_image)

                    cv2.namedWindow('Crop Depth Image',cv2.WINDOW_NORMAL)
                    cv2.imshow("Crop Depth Image", crop_transformed_depth_image)

                #color_image = pyK4A.body_tracker.draw2DSkeleton(skeleton2D_rgb, body.id, color_image)
                #transformed_depth_image = pyK4A.body_tracker.draw2DSkeleton(skeleton2D_rgb, body.id, transformed_depth_image)
                #depth_image = pyK4A.body_tracker.draw2DSkeleton(skeleton2D_depth, body.id, depth_image)

            # Plot the image
            cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
            cv2.imshow("Color Image",color_image)

            cv2.namedWindow('Trans Depth Image',cv2.WINDOW_NORMAL)
            cv2.imshow("Trans Depth Image", transformed_depth_image)

            #cv2.namedWindow('Depth Image',cv2.WINDOW_NORMAL)
            #cv2.imshow("Depth Image",depth_image)

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