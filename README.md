# depthfrommotion
initial:  
frame: 一張初始照片  
width:照片寬度(default = 640)  
length:照片長度(default = 480)  
focalLength:相機焦距(default = 342.75552161810754)  
dfloor:身高(default = 1.6)  
  
使用方法:  
class: depth_from_motion  
function depth_caculation  

input:  
xd: x軸位移量(x軸為正向指向右方)  
yd: y軸位移量(y軸為正向指向下方)  
zd: z軸位移量(z軸為正向指向前方)  
rx: x為轉軸的弧度變化量(右手定則)  
ry: y為轉軸的弧度變化量(右手定則)  
rz: z為轉軸的弧度變化量(右手定則)  
real_rx: x轉軸實際角度(平視0度)  
real_rz: z轉軸實際角度(平視0度)  

output:   
1個五個內容物的array，強度分為0,1,2,3  

注意事項:  
函示裡面有cv2.imshow如果不想要要進去把他刪掉  
## example code  
from depthFromMotion import depth_from_motion  
import numpy as np  
import cv2  
from numpy import linalg as LA  
from scipy.spatial.transform import Rotation as R  
from numba import jit  
import cmapy  
from timeit import default_timer as timer  
cap = cv2.VideoCapture("video.mkv")  
pose = open('pose_add.txt', 'r')  
ret, frame = cap.read()  
prev = frame.copy()  
X = depth_from_motion(prev)  
for line in pose:  
    time, tx_, ty_, tz_, rx_, ry_, rz_, real_rx_, real_rz_ = line.split()  
    xd, yd, zd, rx, ry, rz, real_rx, real_rz = float(tx_), float(ty_), float(tz_), np.radians(float(rx_)), np.radians(float(ry_)), np.radians(float(rz_)),\  
    np.radians(float(real_rx_)), np.radians(float(real_rz_))  
    a = X.depth_caculation(xd, yd, zd, rx, ry, rz, real_rx, real_rz, frame)  
    print(a)  
    ret, frame = cap.read()  
    
    
