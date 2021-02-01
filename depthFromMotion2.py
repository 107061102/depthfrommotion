import numpy as np
import cv2
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from numba import jit
import cmapy
from timeit import default_timer as timer
class depth_from_motion():
    def __init__(self, frame, width = 640, height = 480, focalLength = 342.75552161810754, dfloor = 1.6):
        self.__dfloor = dfloor
        self.__width = width
        self.__height = height
        self.__focalLength = focalLength
        self.__hg = 20
        self.dis = cv2.DISOpticalFlow_create(0)
        self.K = np.array([[focalLength, 0.0, width/2],
             [0.0, focalLength, height/2],
             [0.0, 0.0, 1.0]])
        self.K_inv = LA.inv(self.K)
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.wall = np.loadtxt('wall.txt')
    
    def depth_caculation(self, xd, yd, zd, rx, ry, rz, real_rx, real_rz, frame):
        start = timer()
        frame_count = 0
        time_flow = 0
        time_rt = 0
        time_depth = 0
        time_contour = 0
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_flow = timer()
        diff = cv2.absdiff(self.prev, curr)
        flow = self.dis.calc(self.prev, curr, None, )
        time_flow += timer() - start_flow
        noRotateFlow = np.zeros_like(flow)
        depth = np.zeros_like(curr).astype(np.float32)
        depth += 20
        
        start_rt = timer()
        # rotation in radians
        Rx = R.from_euler('x', -rx).as_matrix()
        Ry = R.from_euler('y', -ry).as_matrix()
        Rz = R.from_euler('z', -rz).as_matrix()
        RI = np.eye(3, 3)
    
        # translation from world frame to camera frame
        translation = np.array([xd, yd, zd]).reshape(3, 1)
        translation_O = [0, 0, 0]
        translation_O = np.array(translation_O).reshape(3, 1)

        # Rt = [R|t]
        Rt_rotate = np.concatenate((Rx @ Ry @ Rz, translation_O), axis=1)
        Rt_translate = np.concatenate((RI, translation), axis=1)
        time_rt += timer() - start_rt
        
        # get depth frame
        start_depth = timer()
        floor = R.from_euler('x', -real_rx).as_matrix()@R.from_euler('z', -real_rz).as_matrix() @ np.array([0, self.__dfloor, 0])
        depth, noRotateFlow = depth_perception(self.wall, self.K, self.K_inv, Rt_rotate, Rt_translate, floor, flow, noRotateFlow, depth, diff, self.__hg)
        time_depth += timer() - start_depth

        # get contours
        start_contour = timer()
        fdepth = (depth < 5) & (depth > 4)
        fdepth = fdepth.astype(np.uint8)
        fdepth = cv2.dilate(fdepth, np.ones((5,5)))
        fdepth = cv2.erode(fdepth, np.ones((3,3)))
        contours, hierarchy = cv2.findContours(fdepth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        fdepth_1 = (depth < 4) & (depth > 3)
        fdepth_1 = fdepth_1.astype(np.uint8)
        fdepth_1 = cv2.dilate(fdepth_1, np.ones((5,5)))
        fdepth_1 = cv2.erode(fdepth_1, np.ones((3,3)))
        contours_1, hierarchy_1 = cv2.findContours(fdepth_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        fdepth_2 = depth < 3
        fdepth_2 = fdepth_2.astype(np.uint8)
        fdepth_2 = cv2.dilate(fdepth_2, np.ones((5,5)))
        fdepth_2 = cv2.erode(fdepth_2, np.ones((3,3)))
        contours_2, hierarchy_2 = cv2.findContours(fdepth_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        time_contour += timer() - start_contour


        depth *= 255 / 20
        depth = 255 - depth
        depth = depth * (depth > 0)
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cmapy.cmap('viridis'))
        cv2.imshow("depth", depth)

    
        showFrame = frame.copy()
        showFrame2 = np.zeros_like(showFrame)
        for i in range(len(contours)):
            if(cv2.contourArea(contours[i]) > 2000):
                cv2.drawContours(showFrame2, contours, i, (0,0,255), -1)
    
        for i in range(len(contours_1)):
            if(cv2.contourArea(contours_1[i]) > 2000):
                cv2.drawContours(showFrame2, contours_1, i, (0,255,0), -1)
    
        for i in range(len(contours_2)):
            if(cv2.contourArea(contours_2[i]) > 2000):
                cv2.drawContours(showFrame2, contours_2, i, (255,0,0), -1)
    
        arr=np.zeros(5)
    
    

        if np.sum(showFrame2[120:400:10,0:60:10,0] == 255)>25:
            cv2.rectangle(showFrame, (0,5), (60,475), (255,0,0), 3)
            arr[0] = 1
        elif np.sum(showFrame2[120:400:10,0:60:10,1] == 255)>25:
            cv2.rectangle(showFrame, (0,5), (60,475), (0,255,0), 3)
            arr[0] = 2
        elif np.sum(showFrame2[120:400:10,0:60:10,2] == 255)>25:
            cv2.rectangle(showFrame, (0,5), (60,475), (0,0,255), 3)
            arr[0] = 3
        
        if  np.sum(showFrame2[120:400:10,60:180:10,0] == 255)>40:
            cv2.rectangle(showFrame, (60,5), (180,475), (255,0,0), 3)
            arr[1] = 1
        elif np.sum(showFrame2[120:400:10,60:180:10,1] == 255)>40:
            cv2.rectangle(showFrame, (60,5), (180,475), (0,255,0), 3)
            arr[1] = 2
        elif np.sum(showFrame2[120:400:10,60:180:10,2] == 255)>40:
            cv2.rectangle(showFrame, (60,5), (180,475), (0,0,255), 3)
            arr[1] = 3
        
        if  np.sum(showFrame2[120:400:10,180:460:10,0] == 255)>50:
            cv2.rectangle(showFrame, (180,5), (460,475), (255,0,0), 3)
            arr[2] = 1
        elif np.sum(showFrame2[120:400:10,180:460:10,1] == 255)>50:
            cv2.rectangle(showFrame, (180,5), (460,475), (0,255,0), 3)
            arr[2] = 2
        elif np.sum(showFrame2[120:400:10,180:460:10,2] == 255)>50:
            cv2.rectangle(showFrame, (180,5), (460,475), (0,0,255), 3)
            arr[2] = 3
        
        if  np.sum(showFrame2[120:400:10,460:580:10,0] == 255)>40:
            cv2.rectangle(showFrame, (460,5), (580,475), (255,0,0), 3)
            arr[3] = 1
        elif np.sum(showFrame2[120:400:10,460:580:10,1] == 255)>40:
            cv2.rectangle(showFrame, (460,5), (580,475), (0,255,0), 3)
            arr[3] = 2
        elif np.sum(showFrame2[120:400:10,460:580:10,2] == 255)>40:
            cv2.rectangle(showFrame, (460,5), (580,475), (0,0,255), 3)
            arr[3] = 3
            
        if  np.sum(showFrame2[120:400:10,580:640:10,0] == 255)>25:
            cv2.rectangle(showFrame, (580,5), (640,475), (255,0,0), 3)
            arr[4] = 1
        elif np.sum(showFrame2[120:400:10,580:640:10,1] == 255)>25:
            cv2.rectangle(showFrame, (580,5), (640,475), (0,255,0), 3)
            arr[4] = 2
        elif np.sum(showFrame2[120:400:10,580:640:10,2] == 255)>25:
            cv2.rectangle(showFrame, (580,5), (640,475), (0,0,255), 3)
            arr[4] = 3
                
        cv2.imshow("obstacles", showFrame)
        cv2.imshow("dot", showFrmae2)
        #writer_obs.write(showFrame)
    
    # keyframe selection (not necessary, uncomment if you want to)
    #if(abs(np.degrees(rx-prx)) > 1 or abs(np.degrees(ry-pry)) > 1 or np.sqrt((tx-ptx)**2+(ty-pty)**2+(tz-ptz)**2) > 0.05):
    #    prev = curr
    #    ptime, ptx, pty, ptz, prx, pry = time, tx, ty, tz, rx, ry
        self.prev = curr
        
        cv2.waitKey(1)
        return(arr)



        
@jit
def depth_perception(wall, K, K_inv, Rt_rotate, Rt_translate, floor, flow, noRotateFlow, depth, diff, hg):
    for point in wall:

        x, y = int(point[0]), int(point[1])
        
        # template wall ideal rotation flow
        [ur, vr, wr] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        ur = ur/wr
        vr = vr/wr
        wr = 1.0

        # template wall translation flow
        [ut, vt, wt] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        ut = ut/wt
        vt = vt/wt
        wt = 1.0
        idealVector = np.array([ut-x, vt-y]).astype(np.float32)
        mag = np.sqrt(idealVector[0]**2 + idealVector[1]**2)
        
        # dynamically generated floor flow
        cfloor = K_inv @ np.array([point[0], point[1], 1])
        r = (floor[0]**2 + floor[1]**2 + floor[2]**2) / (floor[0]*cfloor[0] + floor[1]*cfloor[1] + floor[2]*cfloor[2])
        f = np.array([cfloor[0]*r, cfloor[1]*r, cfloor[2]*r, 1])
        f *= r > 0
        
        for i in range(y-hg,y+hg):
            for j in range(x-hg,x+hg):
                if( mag > 0.2):
                    
                    # rotation compensation
                    noRotateFlow[i,j,0] = flow[i,j,0] - (ur-x)
                    noRotateFlow[i,j,1] = flow[i,j,1] - (vr-y)
                    
                    # depth from motion
                    nRFlow = np.sqrt(noRotateFlow[i,j,0]**2+noRotateFlow[i,j,1]**2)
                    if(nRFlow > 0):
                        depth[i,j] = mag / nRFlow * 20
                        if(np.abs(depth[i,j] - f[2]) < 0.6): depth[i,j] = 20

    return depth, noRotateFlow      
