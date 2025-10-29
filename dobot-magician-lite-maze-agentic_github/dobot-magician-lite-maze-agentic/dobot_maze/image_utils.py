from __future__ import annotations
import cv2, numpy as np
def crop_to_largest_component(img):
    if img.ndim == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]
def mask_markers_and_whiten(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]/179.0; s = hsv[:,:,1]/255.0; v = hsv[:,:,2]/255.0
    green_mask = ((h>0.25)&(h<0.45)&(s>0.2)&(v>0.2)).astype(np.uint8)*255
    red_mask   = (((h<0.05)|(h>0.9))&(s>0.2)&(v>0.2)).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    color_mask = cv2.dilate(cv2.bitwise_or(green_mask, red_mask), kernel, 1)
    img_proc = img_bgr.copy(); img_proc[color_mask>0] = 255
    return img_proc, red_mask, green_mask
