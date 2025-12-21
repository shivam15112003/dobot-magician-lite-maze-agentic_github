# Dobot Magician Lite Maze — Agentic AI Scene + Local Planner (Ubuntu, Camera/File)

Maze solving and execution on **Dobot Magician Lite** using **pydobot**.  
**Agentic AI (Gemini)** is used for the *scene* step; the **path** uses a robust **local Dijkstra planner**.  
Supports **camera capture** (`--camera-index`) *and* **saved images**.

## Features
- **Single input**: `--start-color {red|green}` (other color auto becomes goal).
- **Scene(Agent AI)** terminal print with corners & centroids (values computed locally).
- **Local RAS-style planner** (Dijkstra + smoothing + resample) for the path.
- **Camera support** (interactive preview or headless one-frame capture).
- **Z height**: default work Z is **120 mm**; override with `--z-work`.

## 🎥 Video Demonstration

See the **Agentic AI** and **Local Planner** executing the maze solution on the **Dobot Magician Lite**.

<div align="center">

[![Watch Demo on LinkedIn](https://img.shields.io/badge/Watch_Demo-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/ss1511_asu-robotics-ai-activity-7394842320443072512-O4I-?utm_source=share&utm_medium=member_desktop&rcm=ACoAADOiEBcBfeZohokyDTXwG0fo2BslAbJFwUk)

**[▶️ Click here to view the full activity post](https://www.linkedin.com/posts/ss1511_asu-robotics-ai-activity-7394842320443072512-O4I-?utm_source=share&utm_medium=member_desktop&rcm=ACoAADOiEBcBfeZohokyDTXwG0fo2BslAbJFwUk)**

</div>

## Install
```bash
sudo apt update && sudo apt install -y libgl1 libglib2.0-0
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run (saved image)
```bash
python main.py --image maze8.jpg --start-color red --scene-agent gemini --dryrun --animate
```

## Run (camera)
Interactive preview (press SPACE to capture):
```bash
python main.py --camera-index 0 --start-color green --scene-agent gemini --animate
```
Headless (no preview; auto-capture one frame):
```bash
python main.py --camera-index 0 --camera-auto-capture --no-show --start-color red --scene-agent gemini --dryrun
```
Optional resolution:
```bash
python main.py --camera-index 0 --camera-w 1280 --camera-h 720 --start-color red --dryrun
```

### Terminal prints
```
[Agent] Using agentic AI (Gemini) for scene.
[Agent] Successfully implemented Gemini.
[Scene(Agent AI)] Corners TL=(...) TR=(...) BR=(...) BL=(...)
[Scene(Agent AI)] Red centroid=(...)  Green centroid=(...)
```

## Dobot notes
- Uses `pydobot`; serial port auto-detects (USB/ACM).  
- Always verify `--z-work` clearance before enabling motion.

## License
MIT © 2025 Shivam (shivam15112003)
