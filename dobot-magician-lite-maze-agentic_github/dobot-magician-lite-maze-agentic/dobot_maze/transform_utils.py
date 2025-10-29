from __future__ import annotations
import numpy as np, cv2
def compute_projective_matrix(image_pts_cm, robot_pts_mm):
    src = np.array(list(image_pts_cm), np.float32)
    dst = np.array(list(robot_pts_mm), np.float32)
    return cv2.getPerspectiveTransform(src, dst)
def transform_points(M, pts_cm):
    pts = np.array(pts_cm, np.float32).reshape(-1,1,2)
    return cv2.perspectiveTransform(pts, M).reshape(-1,2)
