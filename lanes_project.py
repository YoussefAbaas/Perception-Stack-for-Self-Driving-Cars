import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import os
import sys

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

