import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import os
import sys

def canny_edge_detector(img):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    v_channel=  hsv[:,:,2]
    _,v_channel=cv2.threshold(v_channel,190, 255, cv2.THRESH_BINARY_INV)
    edged1=cv2.Canny(s_channel,50,150)
    edged2=cv2.Canny(v_channel,50,150)
    edged=edged1 | edged2
    return edged

def perspective_warp(img):
    dst_size=(img.shape[1],img.shape[0])
    src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
    dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img):
    dst_size=(img.shape[1],img.shape[0])
    src=np.float32([(0,0), (1, 0), (0,1), (1,1)])
    dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_curve(img, leftx, rightx):
    # Generate y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    # calculate meters per pixel (field of view in meters / image width)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension 
    xm_per_pix = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #calculate car position
    car_pos = img.shape[1]/2

    #calculate lane center 
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2

    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)



def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

debugging_mode=0
def vid_pipeline(img):
    edged = canny_edge_detector(img)
    warped = perspective_warp(edged)
    curves_image,curves= sliding_window(warped, draw_windows=False)
    curve_radius =get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curve_radius[0], curve_radius[1]])
    img = draw_lanes(img, curves[0], curves[1])
    
    font = cv2.FONT_HERSHEY_PLAIN
    fontColor = (255, 255, 255)
    fontSize=0.9

    cv2.putText(img, 'Radius of Curvature: {:.0f} m'.format(lane_curve), (30, 100), font, fontSize, fontColor, 2)
    if curve_radius[2] < 0:
       cv2.putText(img, 'Vehicle is {:.4f} m left of the center'.format(abs(curve_radius[2])), (30, 130), font, fontSize, fontColor, 2)
    else :
       cv2.putText(img, 'Vehicle is {:.4f} m right of the center'.format(abs(curve_radius[2])), (30, 130), font, fontSize, fontColor, 2)
    if debugging_mode:
    # add stages images to video
        edged=cv2.resize(edged,(200,200))
        edged = np.dstack((edged, edged, edged))*255
        x_offset=1080
        y_offset=0
        img[y_offset:y_offset+edged.shape[0], x_offset:x_offset+edged.shape[1]] = edged

        warped=cv2.resize(warped,(200,200))
        warped = np.dstack((warped, warped, warped))*255
        x_offset=1080
        y_offset=260
        img[y_offset:y_offset+warped.shape[0], x_offset:x_offset+warped.shape[1]] = warped

        curves_image=cv2.resize(curves_image,(200,200))
        x_offset=1080
        y_offset=520
        img[y_offset:y_offset+curves_image.shape[0], x_offset:x_offset+curves_image.shape[1]] = curves_image
    return img
print(sys.version)
debugging_mode=int(sys.argv[1])
videos_list=os.listdir(os.path.dirname(__file__)+'/test_videos')
print(videos_list)
for filename in videos_list:
    if filename.endswith(".mp4"):
        myclip=VideoFileClip(os.path.dirname(__file__)+'/test_videos/'+str(filename)) # Change here
        output_vid = myclip.filename[:-4]+'_output.mp4'
        clip = myclip.fl_image(vid_pipeline)
        clip.write_videofile(output_vid, audio=False)

