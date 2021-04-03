
import sys

from numpy.core.fromnumeric import shape
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
#from kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

def copyto(scr_Image, mask):
	height = scr_Image.shape[0]
	width = scr_Image.shape[1]
	# 生成和原图一样高度和宽度的矩形（全为0）
	dst_Image = scr_Image

	# 以下是copyTo的算法原理：
	# 先遍历每行每列（如果不是灰度图还需遍历通道，可以事先把mask图转为灰度图）
	for row in range(mask.shape[0]):
		for col in range(mask.shape[1]):

			# 如果掩图的像素不等于0，则dst(x,y) = scr(x,y)
			if mask[row, col] != 0:
				# dst_image和scr_Image一定要高宽通道数都相同，否则会报错
				for channel in range(3):
					dst_Image[row, col, channel] = 0
					
			# 如果掩图的像素等于0，则dst(x,y) = 0
			elif mask[row, col] == 0:
				for channel in range(3):
					dst_Image[row, col, channel] = scr_Image[row, col, channel]
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
			for body in pyK4A.body_tracker.bodiesNow:
				pyK4A.body_tracker.printBodyPosition(body)

			# Read and convert the Depth image data to numpy array:
			depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
            # Read and convert the RGB image data to numpy array:
			color_image = pyK4A.image_convert_to_numpy(color_image_handle)

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

			#body_image_color_cv = cv2.fromarray(body_image_color)
			dst_image = copyto(color_image, transformed_custom_image)
			
			#cv2.imshow('Segmented Depth Image',combined_image)
			cv2.imshow('Segmented RGB Image',dst_image)
            
			k = cv2.waitKey(1)

			# Release the image
			pyK4A.image_release(depth_image_handle)
			#pyK4A.image_release(body_image_handle)
			pyK4A.image_release(color_image_handle)
			pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)

		pyK4A.capture_release()
		pyK4A.body_tracker.release_frame()

		if k==27:    # Esc key to stop
			break

	pyK4A.device_stop_cameras()
	pyK4A.device_close()
