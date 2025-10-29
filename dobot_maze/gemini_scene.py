from __future__ import annotations
import json, cv2, numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple
try:
    from google import genai
    from google.genai import types as genai_types
    _HAS = True
except Exception:
    _HAS = False
@dataclass
class SceneDetections:
    corners_px: np.ndarray
    red_centroid: Tuple[int,int]
    green_centroid: Tuple[int,int]
    raw_json: Dict[str, Any]
    model: str
class GeminiSceneError(RuntimeError):
    pass
PROMPT = """You are a vision agent. Given a maze image (light corridors, dark walls) with a RED start dot and a GREEN goal dot:
Return STRICT JSON only:
{ "corners": { "tl":[x,y], "tr":[x,y], "br":[x,y], "bl":[x,y] },
  "red_centroid": [x,y],
  "green_centroid": [x,y] }
Coordinates must be integers and inside the image; corners ordered TL,TR,BR,BL.
"""
def _encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok: raise GeminiSceneError("PNG encode failed")
    return buf.tobytes()
def _order_corners(c, shape):
    h, w = shape[:2]
    def clamp(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return [max(0, min(w-1, x)), max(0, min(h-1, y))]
    tl = clamp(c.get("tl", [0,0]))
    tr = clamp(c.get("tr", [w-1,0]))
    br = clamp(c.get("br", [w-1,h-1]))
    bl = clamp(c.get("bl", [0,h-1]))
    return np.array([tl,tr,br,bl], dtype=np.float32)
def _centroid(v, shape):
    h,w = shape[:2]
    x = max(0, min(w-1, int(round(v[0]))))
    y = max(0, min(h-1, int(round(v[1]))))
    return (x,y)
def detect_scene_with_gemini(img_bgr: np.ndarray, model: str = "gemini-2.5-flash", timeout: float = 30.0) -> SceneDetections:
    if not _HAS: raise GeminiSceneError("google-genai not installed. pip install google-genai")
    client = genai.Client()
    cfg = genai_types.GenerateContentConfig(response_mime_type="application/json")
    resp = client.models.generate_content(
        model=model,
        contents=[genai_types.Part.from_text(PROMPT), genai_types.Part.from_bytes(b=_encode_png(img_bgr), mime_type="image/png")],
        config=cfg
    )
    try:
        data = json.loads(resp.text if hasattr(resp, "text") else str(resp))
    except Exception as e:
        raise GeminiSceneError(f"Invalid JSON: {e}")
    corners = _order_corners(data.get("corners", {}), img_bgr.shape)
    rc = _centroid(data.get("red_centroid", [0,0]), img_bgr.shape)
    gc = _centroid(data.get("green_centroid", [0,0]), img_bgr.shape)
    return SceneDetections(corners, rc, gc, data, model)
def local_detect_corners(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h,w = gray.shape
        return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        x,y,w,h = cv2.boundingRect(cnt)
        pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)
def local_detect_centroids(red_mask: np.ndarray, green_mask: np.ndarray):
    def centroid(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0: return None
        return (int(np.mean(xs)), int(np.mean(ys)))
    return centroid(red_mask), centroid(green_mask)
