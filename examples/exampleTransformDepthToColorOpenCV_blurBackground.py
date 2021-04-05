import sys
sys.path.insert(0, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a, postProcessing
import cv2
import time

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

def alphaBlend(img1, img2, mask):
    """ 
    alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0

    # 此處僅使用convertScaleAbs函數輸出為unit8的功能
    # 等同於(img1*(1-alpha) + img2*alpha).astype('uint8')
    # 若非uint8格式，將無法正常顯示畫面
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

if __name__ == "__main__":

    dis_near = 200
    dis_far = 900

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Open device
    pyK4A.device_open()

    # Modify camera configuration
    device_config = pyK4A.config
    device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)

    front_ground = np.zeros((1280,720))
    front_ground_mask = np.zeros((1280,720))
    back_ground = np.zeros((1280,720))
    back_ground_mask = np.zeros((1280,720))

    k = 0
    time_start = time.time()
    cv2.namedWindow('Output Image',cv2.WINDOW_NORMAL)

    while True:
        # Get capture
        pyK4A.device_get_capture()

        # Get the depth image from the capture
        depth_image_handle = pyK4A.capture_get_depth_image()

        # Get the color image from the capture
        color_image_handle = pyK4A.capture_get_color_image()

        

        # Check the image has been read correctly
        if depth_image_handle and color_image_handle:

            # Read and convert the image data to numpy array:
            color_image = pyK4A.image_convert_to_numpy(color_image_handle)[:,:,:3]

            front_ground = color_image.copy()
            back_ground = color_image.copy()

            # Transform the depth image to the color format
            transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle,color_image_handle)

            maximum_hole_size = 10
            smoothed_depth_image = postProcessing.smooth_depth_image(transformed_depth_image, maximum_hole_size)
            #print(smoothed_depth_image.shape)

            #height, width = color_image.shape[:2]

            mask = smoothed_depth_image > dis_near
            #print(mask.shape)
            front_ground[mask] = 0

            back_ground[~mask] = 0
            #print(back_ground.dtype)
            #front_ground_mask = 1*mask
            back_ground_mask = (255*mask).astype('uint8')
            #print(back_ground_mask.dtype)

            back_ground = cv2.GaussianBlur(color_image, (21,21),11 )
            back_ground_mask = cv2.GaussianBlur(back_ground_mask, (21,21),11 )

            output_image = alphaBlend(color_image, back_ground, back_ground_mask)
            #output_image = front_ground+back_ground


            # Convert depth image (mm) to color, the range needs to be reduced down to the range (0,255)
            #transformed_depth_color_image = cv2.applyColorMap(np.round(transformed_depth_image/30).astype(np.uint8), cv2.COLORMAP_JET)

            # Add the depth image over the color image:
            #combined_image = cv2.addWeighted(color_image,0.7,transformed_depth_color_image,0.3,0)

            fps = int(60/(time.time()-time_start))

            cv2.putText(output_image, "FPS: "+str(fps),(50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 0, 255), 1, cv2.LINE_AA)
            
            # Plot the image
            
            cv2.imshow('Output Image',output_image)

            time_start = time.time()
            #cv2.imshow('Depth Mask',back_ground_mask)
            k = cv2.waitKey(10)

            pyK4A.image_release(depth_image_handle)
            pyK4A.image_release(color_image_handle)

        pyK4A.capture_release()

        # Esc key to stop
        if k==27:
            cv2.destroyAllWindows()
            break

    pyK4A.device_stop_cameras()
    pyK4A.device_close()