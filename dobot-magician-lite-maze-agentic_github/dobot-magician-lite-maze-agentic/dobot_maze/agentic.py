from __future__ import annotations
import cv2, numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
from .path_plan import dijkstra_path, gaussian_smooth_path, interparc_linear, snap_to_safe
@dataclass
class PlanResult:
    walk_mask: np.ndarray
    dist_map: np.ndarray
    raw_path_px: np.ndarray
    smooth_path_px: np.ndarray
    resampled_path_px: np.ndarray
    score: float
    metrics: Dict[str, float]
    params: Dict[str, Any]
class AgenticPlanner:
    def __init__(self, start_xy: Tuple[int,int], goal_xy: Tuple[int,int]):
        self.start_xy = start_xy
        self.goal_xy = goal_xy
        self.log: List[str] = []
    def _binarize(self, img_bgr: np.ndarray, block_size: int, C: int, close_ksize: int) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bs = block_size if block_size % 2 == 1 else block_size + 1
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C)
        if close_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
        return (bw > 0).astype(np.uint8)
    def _distance(self, walk_mask: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform(walk_mask, cv2.DIST_L2, 3)
    def _evaluate(self, path_px: np.ndarray, dist_map: np.ndarray):
        if len(path_px)==0: return dict(length=np.inf, min_clearance=0.0, mean_clearance=0.0, score=np.inf)
        idx = path_px[:,1].astype(int), path_px[:,0].astype(int)
        clear = dist_map[idx]; L=float(len(path_px))
        min_c=float(clear.min()) if clear.size else 0.0
        mean_c=float(clear.mean()) if clear.size else 0.0
        score= L - 4.0*min_c - 1.0*mean_c
        return dict(length=L, min_clearance=min_c, mean_clearance=mean_c, score=score)
    def plan(self, img_bgr_proc: np.ndarray) -> PlanResult:
        cands=[
            dict(block_size=31, C=2, close_ksize=0),
            dict(block_size=25, C=2, close_ksize=0),
            dict(block_size=35, C=2, close_ksize=0),
            dict(block_size=31, C=4, close_ksize=0),
            dict(block_size=31, C=2, close_ksize=3),
            dict(block_size=31, C=2, close_ksize=5),
        ]
        best: Optional[PlanResult] = None
        for p in cands:
            walk = self._binarize(img_bgr_proc, **p); dist = self._distance(walk)
            s,_ = snap_to_safe(self.start_xy, walk.astype(bool)); g,_ = snap_to_safe(self.goal_xy, walk.astype(bool))
            _,_,raw = dijkstra_path(walk.astype(bool), dist, s, g)
            smooth = gaussian_smooth_path(raw.astype(np.float64), 4.0) if len(raw) else raw
            res = interparc_linear(20, smooth[:, :2]) if len(smooth) else smooth
            m = self._evaluate(raw, dist); sc=m['score']
            self.log.append(f"params={p} -> found={len(raw)>0} score={sc:.2f} len={m['length']:.0f} minC={m['min_clearance']:.2f}")
            if best is None or sc < best.score:
                best = PlanResult(walk, dist, raw, smooth, res, sc, m, p)
            if len(raw)>0 and m['min_clearance']>=2.0: break
        if best is None or len(best.raw_path_px)==0:
            raise RuntimeError("Planner could not find a valid path.")
        return best
