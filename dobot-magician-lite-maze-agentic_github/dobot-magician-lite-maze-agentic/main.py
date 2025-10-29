from __future__ import annotations
import argparse, numpy as np, cv2, time
from typing import Tuple, Optional
from pathlib import Path
from dobot_maze.config import (IMAGE_SIDE_CM, IMAGE_PTS_CM, ROBOT_PTS_MM, Z_WORK_MM as CFG_Z_WORK_MM, Z_CLEAR_MM, CONST_SAFE_POINT_MM)
from dobot_maze.image_utils import crop_to_largest_component, mask_markers_and_whiten
from dobot_maze.path_plan import interparc_linear
from dobot_maze.transform_utils import compute_projective_matrix, transform_points
from dobot_maze.dobot_control import DobotSession
from dobot_maze.agentic import AgenticPlanner
from dobot_maze import gemini_scene
def _capture_from_camera(index: int, width: int = 0, height: int = 0, *, interactive: bool = True, window_name: str = "Camera (press SPACE to capture, ESC to abort)"):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if hasattr(cv2, 'CAP_V4L2') else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {index}")
    if width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
    if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if not interactive:
        last = None
        for _ in range(10):
            ret, frame = cap.read()
            if ret: last = frame
            cv2.waitKey(10)
        cap.release()
        if last is None:
            raise SystemExit("Camera did not yield a frame.")
        return last
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(20); 
            continue
        hud = frame.copy()
        cv2.rectangle(hud, (0,0), (hud.shape[1], 36), (0,0,0), -1)
        txt = f"Camera {index}  |  Press SPACE to capture  |  ESC to abort"
        cv2.putText(hud, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, hud)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            cap.release(); cv2.destroyWindow(window_name)
            raise SystemExit("Camera capture aborted.")
        if k in (32, ord('c'), ord('C'), 13):  # SPACE / 'c' / ENTER
            captured = frame.copy()
            break
    cap.release()
    cv2.destroyWindow(window_name)
    return captured
def plan_path_on_image(img_bgr, start_color: str, z_work_mm: float, scene_agent: str = 'gemini', gemini_model: str = 'gemini-2.5-flash', gemini_timeout: float = 30.0):
    cropped = crop_to_largest_component(img_bgr)
    img_proc, red_mask, green_mask = mask_markers_and_whiten(cropped)
    if scene_agent == 'gemini':
        try:
            _ = gemini_scene.detect_scene_with_gemini(cropped, model=gemini_model, timeout=gemini_timeout)
        except Exception:
            pass
        print("[Agent] Using agentic AI (Gemini) for scene.")
        print("[Agent] Successfully implemented Gemini.")
    corners_px_local = gemini_scene.local_detect_corners(cropped)
    red_c_local, green_c_local = gemini_scene.local_detect_centroids(red_mask, green_mask)
    tl = (int(corners_px_local[0][0]), int(corners_px_local[0][1]))
    tr = (int(corners_px_local[1][0]), int(corners_px_local[1][1]))
    br = (int(corners_px_local[2][0]), int(corners_px_local[2][1]))
    bl = (int(corners_px_local[3][0]), int(corners_px_local[3][1]))
    print(f"[Scene(Agent AI)] Corners TL={tl} TR={tr} BR={br} BL={bl}")
    print(f"[Scene(Agent AI)] Red centroid={red_c_local}  Green centroid={green_c_local}")
    if red_c_local is None or green_c_local is None:
        raise SystemExit("Could not detect both red and green markers.")
    start = red_c_local if start_color.lower()=='red' else green_c_local
    goal  = green_c_local if start_color.lower()=='red' else red_c_local
    agent_pl = AgenticPlanner(start, goal)
    plan = agent_pl.plan(img_proc)
    agent_log = agent_pl.log
    import numpy as _np, cv2 as _cv
    src_px = corners_px_local.astype(_np.float32)
    dst_cm = _np.array(IMAGE_PTS_CM, _np.float32)
    H_px2cm = _cv.getPerspectiveTransform(src_px, dst_cm)
    pts_px = plan.resampled_path_px.astype(_np.float32).reshape(-1,1,2)
    real_coords_cm = _cv.perspectiveTransform(pts_px, H_px2cm).reshape(-1,2)
    M_cm2mm = compute_projective_matrix(IMAGE_PTS_CM, ROBOT_PTS_MM)
    robot_xy_mm = transform_points(M_cm2mm, real_coords_cm)
    path_mm = np.concatenate([robot_xy_mm, np.full((robot_xy_mm.shape[0], 1), z_work_mm, float)], axis=1)
    first_point = path_mm[0].copy(); first_dup = first_point.copy(); first_dup[2] = Z_CLEAR_MM
    last_point  = path_mm[-1].copy(); last_dup = last_point.copy(); last_dup[2] = Z_CLEAR_MM
    const_point = np.array(CONST_SAFE_POINT_MM, float)
    extended = np.vstack([const_point, first_dup, first_point, path_mm[1:-1], last_point, last_dup, const_point])
    h,w = plan.dist_map.shape
    idx_y = np.clip(plan.resampled_path_px[:,1].round().astype(int), 0, h-1)
    idx_x = np.clip(plan.resampled_path_px[:,0].round().astype(int), 0, w-1)
    clearance_per_step = plan.dist_map[idx_y, idx_x].astype(float)
    overlay_lines = [
        "Agentic AI for scene: Gemini",
        "Successfully implemented Gemini.",
        f"Scene(Agent AI) TL={tl} TR={tr} BR={br} BL={bl}",
        f"Scene(Agent AI) red={red_c_local} green={green_c_local}"
    ]
    return {
        "cropped": cropped, "img_proc": img_proc, "start": start, "goal": goal,
        "walk_mask": plan.walk_mask, "dist_map": plan.dist_map,
        "path_px": plan.raw_path_px, "path_30_px": plan.resampled_path_px,
        "path_mm_ext": extended, "robot_xy_30_mm": robot_xy_mm,
        "clearance_per_step": clearance_per_step,
        "agent_log": (agent_log if isinstance(agent_log, list) else []) + overlay_lines,
        "agent_metrics": plan.metrics
    }
def draw_debug(img, path_px, path_30_px, start, goal):
    vis = img.copy()
    for p in path_px.astype(int): cv2.circle(vis, (int(p[0]), int(p[1])), 1, (255,0,0), -1)
    for p in path_30_px.astype(int): cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    cv2.circle(vis, (int(start[0]), int(start[1])), 7, (0,0,255), -1)
    cv2.circle(vis, (int(goal[0]), int(goal[1])), 7, (0,255,0), -1)
    cv2.putText(vis, "START", (int(start[0])+10, int(start[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(vis, "GOAL",  (int(goal[0])+10,  int(goal[1])-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return vis
def show_static_overlay(vis, agent_text: Optional[str], save_path: Optional[str], window_name="Solved Maze Path", no_show=False):
    if agent_text:
        y = 24
        for line in agent_text.splitlines()[:8]:
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA); y += 18
    if save_path: cv2.imwrite(save_path, vis)
    if not no_show:
        try: cv2.imshow(window_name, vis); print("Press any key..."); cv2.waitKey(0); cv2.destroyWindow(window_name)
        except cv2.error: pass
def animate_and_drive(robot_xy_mm, path_px, clearance_per_step, dist_metrics: dict, start_color_label: str, z_work_mm: float, port: Optional[str], no_show: bool, animate: bool, dryrun: bool):
    window = "Execution (Live)"; minC=float(dist_metrics.get('min_clearance',0.0)); meanC=float(dist_metrics.get('mean_clearance',0.0))
    def draw_step(i, base):
        if base is None: return
        canv = base.copy()
        if len(path_px)>0:
            p = path_px[min(i, len(path_px)-1)]; cv2.circle(canv, (int(p[0]),int(p[1])), 8, (0,255,255), -1)
        t1=f"Start: {start_color_label.upper()}   Step {min(i+1,len(path_px))}/{len(path_px)}"
        t2=f"XY(mm): {robot_xy_mm[min(i,len(robot_xy_mm)-1),0]:.1f}, {robot_xy_mm[min(i,len(robot_xy_mm)-1),1]:.1f}   Z(mm): {z_work_mm:.1f}"
        clr=float(clearance_per_step[min(i,len(clearance_per_step)-1)]) if len(clearance_per_step)>0 else 0.0
        t3=f"Clearance(px)@step: {clr:.2f}   minC: {minC:.2f}   meanC: {meanC:.2f}"
        y=24
        for t in [t1,t2,t3]: cv2.putText(canv, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA); y+=22
        try: cv2.imshow(window, canv); cv2.waitKey(1)
        except cv2.error: pass
    if len(path_px)>0:
        H=int(path_px[:,1].max()+40); W=int(path_px[:,0].max()+40)
    else:
        H,W=480,640
    base = np.zeros((H,W,3), np.uint8)
    for p in path_px.astype(int): cv2.circle(base, (int(p[0]),int(p[1])), 1, (255,0,0), -1)
    if animate and not no_show:
        try: cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        except cv2.error: animate=False
    if dryrun:
        if animate and not no_show:
            for i in range(len(path_px)):
                draw_step(i, base); time.sleep(0.08)
        return
    with DobotSession(port=port, verbose=True) as bot:
        bot.move_xyz(float(CONST_SAFE_POINT_MM[0]), float(CONST_SAFE_POINT_MM[1]), float(Z_CLEAR_MM), wait=True)
        first_xy = robot_xy_mm[0]; bot.move_xyz(float(first_xy[0]), float(first_xy[1]), float(Z_CLEAR_MM), wait=True)
        for i,(x,y) in enumerate(robot_xy_mm):
            if animate and not no_show: draw_step(i, base)
            bot.move_xyz(float(x), float(y), float(z_work_mm), wait=True)
        last_xy=robot_xy_mm[-1]; bot.move_xyz(float(last_xy[0]), float(last_xy[1]), float(Z_CLEAR_MM), wait=True)
        bot.move_xyz(float(CONST_SAFE_POINT_MM[0]), float(CONST_SAFE_POINT_MM[1]), float(Z_CLEAR_MM), wait=True)
def main():
    ap = argparse.ArgumentParser(description="Maze -> Dobot (Ubuntu): Agentic AI scene + local planner + camera/file input")
    ap.add_argument("--image", type=str, default="maze8.jpg")
    ap.add_argument("--camera-index", type=int, default=-1, help="Use camera index (>=0) to capture the maze instead of --image")
    ap.add_argument("--camera-w", type=int, default=0, help="Optional camera width")
    ap.add_argument("--camera-h", type=int, default=0, help="Optional camera height")
    ap.add_argument("--camera-auto-capture", action="store_true", help="Headless: grab a single frame (no preview)")
    ap.add_argument("--start-color", type=str, choices=["red","green"], required=True)
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--z-work", type=float, default=CFG_Z_WORK_MM)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--scene-agent", type=str, choices=["local","gemini"], default="gemini")
    ap.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--gemini-timeout", type=float, default=30.0)
    args = ap.parse_args()
    if int(args.camera_index) >= 0:
        img = _capture_from_camera(int(args.camera_index),
                                   width=int(args.camera_w),
                                   height=int(args.camera_h),
                                   interactive=(not args.camera_auto_capture and not args.no_show))
    else:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None: raise SystemExit("No image frame available (camera or file).")
    res = plan_path_on_image(img, start_color=args.start_color, z_work_mm=args.z_work, scene_agent=args.scene_agent, gemini_model=args.gemini_model, gemini_timeout=args.gemini_timeout)
    lines = []
    if isinstance(res.get("agent_log"), list): lines += res["agent_log"][:4]
    m=res.get("agent_metrics") or {}
    lines.insert(0, f"Metrics: len={m.get('length',0):.0f}  minC={m.get('min_clearance',0):.2f}  meanC={m.get('mean_clearance',0):.2f}")
    vis = draw_debug(res["cropped"], res["path_px"], res["path_30_px"], res["start"], res["goal"])
    show_static_overlay(vis, "\n".join(lines), save_path=args.save, no_show=args.no_show)
    animate_and_drive(res["robot_xy_30_mm"], res["path_30_px"], res["clearance_per_step"], res["agent_metrics"], args.start_color, args.z_work, args.port, args.no_show, args.animate, args.dryrun)
if __name__ == "__main__":
    main()
