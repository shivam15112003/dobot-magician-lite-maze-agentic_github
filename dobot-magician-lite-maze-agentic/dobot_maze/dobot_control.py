from __future__ import annotations
from serial.tools import list_ports
import pydobot
from .config import DEFAULT_R_DEG, DEFAULT_VELOCITY, DEFAULT_ACCELERATION, DWELL_MS_BETWEEN_MOVES
CANDIDATE_VID_PID = {(0x1A86,0x7523):"CH340",(0x10C4,0xEA60):"CP210x",(0x067B,0x2303):"Prolific"}
def _auto_detect_port():
    ports = list_ports.comports(); pref=[]
    for p in ports:
        vid,pid=getattr(p,'vid',None),getattr(p,'pid',None)
        if vid is not None and pid is not None and (vid,pid) in CANDIDATE_VID_PID: pref.append(p.device)
    if not pref: pref=[p.device for p in ports if str(p.device).startswith(('/dev/ttyUSB','/dev/ttyACM'))]
    if not pref and ports: pref=[ports[0].device]
    return pref[0] if pref else None
class DobotSession:
    def __init__(self, port: str|None=None, verbose: bool=True):
        if port is None or (isinstance(port,str) and port.lower()=='auto'): port=_auto_detect_port()
        if port is None: raise RuntimeError("No serial ports found for Dobot.")
        if verbose: print(f"[dobot] Using serial port: {port}")
        self.device = pydobot.Dobot(port=port, verbose=verbose)
        self.device.speed(DEFAULT_VELOCITY, DEFAULT_ACCELERATION)
    def pose(self): return self.device.pose()
    def move_xyz(self, x,y,z, r: float=DEFAULT_R_DEG, wait: bool=True):
        self.device.move_to(x,y,z,r,wait=wait)
        if DWELL_MS_BETWEEN_MOVES>0: self.device.wait(DWELL_MS_BETWEEN_MOVES)
    def suck(self, enable: bool): self.device.suck(enable)
    def grip(self, enable: bool): self.device.grip(enable)
    def close(self):
        try: self.device.close()
        except Exception: pass
    def __enter__(self): return self
    def __exit__(self, a,b,c): self.close()
