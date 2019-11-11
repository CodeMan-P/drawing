import cv2 as cv
import numpy as np

import numpy as np

import math
from painter import Painter

from PIL import Image, ImageDraw, ImageFont

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
    
def creatImg(s,width=400):
    mask = np.full((width,width,3),255,np.uint8)
    pilimg = Image.fromarray(mask)
    draw = ImageDraw.Draw(pilimg) # 图片上打印
    fontSize = 50
    while 1:
        font = ImageFont.truetype("simkai.ttf", fontSize, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
        w, h = draw.textsize(s, font=font)
        if max(w,h)<width:
            fontSize+=1
        else:
            fontSize-=1
            break
    font = ImageFont.truetype("simkai.ttf", fontSize, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    w, h = draw.textsize(s, font=font)
    w = max(w,h)
    draw.text((width/2 -w/2, width/2 -w/2), s, (0, 0, 0), font=font)
    mask=np.array(pilimg)
    cv.imwrite('tmp.jpg',mask);
    return 'tmp.jpg'
def toPng(s,radius = 500):
    img = cv.imread(s, cv.IMREAD_UNCHANGED) 
    
    tmpFrame = np.full((img.shape[0],img.shape[1],4),0,np.uint8)
    tmpFrame[:,:,0]=img[:,:]
    tmpFrame[:,:,1]=img[:,:]
    tmpFrame[:,:,2]=img[:,:]
    
    mask = np.full((img.shape[0],img.shape[1]),0,np.uint8)
    cv.circle(mask, (radius,radius), radius, (255,255,255), -1)
    tmpFrame[:,:,3]=mask

    cv.imwrite(s+'.png', tmpFrame, [int(cv.IMWRITE_PNG_COMPRESSION), 3])
def main():
    toPng("ret.jpg")
    s = "裴"
    s = creatImg(s,600)
    
    #mask = cv.putText(mask, s, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    p = Painter(s,500,1)
    #p = Painter('1.png',400,1)

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
