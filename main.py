import cv2 as cv
import numpy as np

import numpy as np

import math
from painter import Painter
width = 600
r= 200
center = (int(width/2),int(width/2))
def getCirclePoints():
    points = []
    for i in range(0,360,2):
        x = center[0] + r * math.cos(i * 3.14 / 180.0)
        y = center[1] + r * math.sin(i * 3.14 / 180.0)
        points.append((int(x),int(y)))
    return points
    
    
def main():
    p = Painter('1.png',400,1)

    #p = Painter('2.jpg',200,1)
    # img=np.full((width,width,3),0,np.uint8)

    # cv.circle(img,center, 250, (255,255,255), -1)
    
    # points =getCirclePoints()
    
    # # for p in points:
    # #     cv.circle(img,p, 1, (0,0,255), 1)
    # # for i in range(0,180,5):
    # #     p1 = points[i]
    # #     interval = 80
    # #     index = i+interval
    # #     if i+interval >= len(points):
    # #         index -= len(points)
    # #     cv.line(img,p1,points[index],(0,0,255),1)
    
    p.run()
    # for x in range(30,100,1):
    #     print(x / 100)
    #     p.show(x / 100)
    #     cv.waitKey(0)
    
    # cv.imshow('opencv',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
if __name__ == '__main__':
    # img=cv.imread('1.png',cv.IMREAD_COLOR)
    # print(type(img))
    # img= cv.resize(img,(width,width))
    # cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
    # cv.imshow('lena',img)
    
    #k=cv.waitKey(0)
    main()
