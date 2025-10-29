from __future__ import annotations
import heapq, numpy as np, cv2
from .config import WALL_PENALTY_GAIN
def binarize_and_distance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    walk = (bw>0).astype(np.uint8)
    dist = cv2.distanceTransform(walk, cv2.DIST_L2, 3)
    return walk, dist
def snap_to_safe(pt, mask):
    x,y = int(pt[0]), int(pt[1]); h,w = mask.shape
    x = np.clip(x,0,w-1); y = np.clip(y,0,h-1)
    if mask[y,x]: return (x,y), False
    ys,xs = np.nonzero(mask)
    if len(xs)==0: return (x,y), False
    idx = np.argmin((xs-x)**2+(ys-y)**2)
    return (int(xs[idx]), int(ys[idx])), True
def dijkstra_path(mask, dist_map, start, goal):
    h,w = mask.shape; sx,sy = start; gx,gy = goal
    dist = np.full((h,w), np.inf); prev = np.full((h,w,2), -1, np.int32); vis = np.zeros((h,w), bool)
    pq=[]; dist[sy,sx]=0.0; heapq.heappush(pq,(0.0,sx,sy))
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    while pq:
        d,x,y = heapq.heappop(pq)
        if vis[y,x]: continue
        vis[y,x]=True
        if x==gx and y==gy: break
        for dx,dy in dirs:
            nx,ny=x+dx,y+dy
            if nx<0 or ny<0 or nx>=w or ny>=h: continue
            if not mask[ny,nx]: continue
            penalty = WALL_PENALTY_GAIN/(dist_map[ny,nx]+1.0)
            nd=d+1.0+penalty
            if nd<dist[ny,nx]:
                dist[ny,nx]=nd; prev[ny,nx]=(x,y); heapq.heappush(pq,(nd,nx,ny))
    path=[]; x,y=gx,gy
    if prev[y,x,0]==-1 and not (gx==sx and gy==sy): return prev,dist,np.empty((0,2),np.int32)
    while not (x==sx and y==sy):
        path.append((x,y)); px,py=prev[y,x]; 
        if px==-1: break
        x,y=int(px),int(py)
    path.append((sx,sy)); path.reverse()
    return prev,dist,np.array(path,np.int32)
def gaussian_smooth_path(P, sigma_px=4.0):
    if len(P)==0: return P
    import numpy as np
    r=int(max(1, round(3*sigma_px))); x=np.arange(-r, r+1)
    k=np.exp(-(x**2)/(2*sigma_px**2)); k/=k.sum()
    pad=np.pad(P,((r,r),(0,0)),'edge'); out=np.empty_like(P,float)
    for d in range(2): out[:,d]=np.convolve(pad[:,d],k,mode='valid')
    return out
def interparc_linear(n, P):
    if len(P)==0: return P
    import numpy as np
    P=P.astype(float); D=np.diff(P,axis=0); L=np.sqrt((D**2).sum(1))
    A=np.concatenate([[0.0], np.cumsum(L)]); T=A[-1]
    if T==0: return np.repeat(P[:1], n, 0)
    targets=np.linspace(0.0,T,n); out=np.empty((n,2),float)
    j=0
    for i,t in enumerate(targets):
        while j<len(A)-1 and A[j+1]<t: j+=1
        if j==len(A)-1: out[i]=P[-1]
        else:
            t0,t1=A[j],A[j+1]
            a=0.0 if t1==t0 else (t-t0)/(t1-t0)
            out[i]=(1-a)*P[j]+a*P[j+1]
    return out
