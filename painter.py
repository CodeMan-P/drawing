from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb

import cv2 as cv
import numpy as np
import math
class Painter:
    radius = 250
    width = 500
    margin = 50
    srcImg = []
    canvas = []
    center = []
    board=[]
    points=[]
    def img2cols(self, img):
        img = img.reshape(img.size, order="C")
        # convert the data type as np.float64
        img = img.astype(np.float64)
        return img
    def lbp(self,img):
        # settings for LBP
        radius = 3  # LBP算法中范围半径的取值
        n_points = 8 * radius # 领域像素点数

        lbp = local_binary_pattern(img, n_points, radius)
        cv.imshow('lbp',lbp)
        return lbp

    def __init__(self, path,radius = 250,showSrc = 0,margin=50):
        self.srcImg=cv.imread(path,cv.IMREAD_COLOR)
        
        if type(None) == type(self.srcImg):
            print('readImg failed!')
            return
        
        self.radius = radius
        
        tmpr = max(self.srcImg.shape)/2
        if(tmpr>self.radius):
            w = int(self.srcImg.shape[0] * self.radius / tmpr)
            h = int(self.srcImg.shape[1]* self.radius / tmpr)
            self.srcImg= cv.resize(self.srcImg,(w,h))
        


        self.margin = margin
        self.width = (radius+self.margin)*2
        
        if showSrc:
            cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
            cv.imshow('lena',self.srcImg)
        self.initCanvas()
    def initCanvas(self):
        width = self.width
        
        self.center = (int(width/2),int(width/2))
        self.canvas=np.full((width,width,3),255,np.uint8)
        mask = np.full((width,width,3),0,np.uint8)
        cv.circle(mask, self.center, self.radius, (255,255,255), -1)
        
        w = self.srcImg.shape[0]
        h = self.srcImg.shape[1]
        #50:50,50+self.radius:50+self.radius
        x1 =int(width/2 - w/2)
        x2 =int(width/2 + w/2)
        y1 =int(width/2 - h/2)
        y2 =int(width/2 + h/2)
        
        cv.copyTo(self.srcImg,mask[x1:x2,y1:y2],self.canvas[x1:x2,y1:y2])
        
        x1 =int(width/2 - self.radius)
        y1 =int(width/2 + self.radius)
        

        #self.board = cv.cvtColor(self.canvas[x1:y1,x1:y1].copy(), cv.COLOR_RGB2GRAY)
        
        self.board= cv.cvtColor(self.canvas[x1:y1,x1:y1], cv.COLOR_RGB2GRAY)
        #self.board= self.canvas[x1:y1,x1:y1].copy()
        #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #self.board = self.lbp(self.board)
        
        self.board[self.board >100] = 255
        self.board[self.board <=100] = 0
        self.board[self.board == 255] = 2
        self.board[self.board == 0] = 1
        self.board[self.board == 2] = 0

        #self.board-=255
        #self.board[self.board < 0] = 255
      
        # shape=self.board.shape
        # for x in range(0, shape[0]):
        #     for y in range(0, shape[1]):
        #         if self.board[x, y] == 255:
        #             self.board[x, y] = 0
        #         else:
        #             self.board[x, y] = 255
      
        # self.board[self.board == 1] = 2
        # self.board[self.board == 0] = 255
        # self.board[self.board == 2] = 0
        self.center = [int(x/2) for x in self.board.shape[:2]]
        self.initCirclePoints()
    
        # for p in self.points:
        #     cv.circle(self.board,p, 1, 1, 1)

        cv.namedWindow('canvas',cv.WINDOW_AUTOSIZE)
        cv.imshow('canvas',self.canvas)
        cv.imshow('board',self.board)
        
        cv.waitKey(30)
        cv.waitKey(0)
        
        
        
    def initCirclePoints(self):
        self.points = []
        r = self.radius
        for i in range(0,360,1):
            x = self.center[0] + r * math.cos(i * 3.14 / 180.0)
            y = self.center[1] + r * math.sin(i * 3.14 / 180.0)
            self.points.append((int(x),int(y)))
    def show(self,offset=0.1):
        tmpCnts = self.tmpCnts.copy()
        tmpmax=self.tmpmax
        results = []
        points = self.points
        mask = np.zeros(self.board.shape)

        flag = 1
        while flag:
            
            for i in range(0,180):
                flag = 0
                tmpMaxOff = 0
                p1 = 0
                p2= 0
                for j in range(0,180):
                    if i==j:
                        continue
                    if tmpCnts[i][j] > tmpmax*offset:
                        if tmpCnts[i][j]>tmpMaxOff:
                            tmpMaxOff = tmpCnts[i][j]
                            tmpCnts[i][j]=0
                            
                            flag = 1
                            p1 = points[i]
                            p2 = points[j]
                            #cv.line(self.board,p1,p2,0,1)
                            #results.append((p1,p2))
                            #print((p1,p2)," ",tmpCnts[i][j]," ",tmpmax)
                if flag:
                    cv.line(mask,p1,p2,255,1)
                    cv.imshow('mask',mask)
                    cv.waitKey(30)
                print(i)
        
        
        
        # if results:
        #     for p1,p2 in results:
        #         cv.line(mask,p1,p2,255,1)
        # cv.imshow('mask',mask)
        
    def run(self):
        ## opening videocapture
        #cap = cv.VideoCapture(0)
        
        ## some videowriter props
        # sz = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        #         int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
        fps = 25
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        #fourcc = cv.VideoWriter_fourcc('m', 'p', 'e', 'g')
        #fourcc = cv.VideoWriter_fourcc(*'mpeg')
        
        ## open and set props
        vout = cv.VideoWriter()
        print(self.board.shape)
        vout.open('./output.mp4',fourcc,fps,self.board.shape,True)

        print('run')
        points = self.points
        
        
        tmpmax = 0
        board = self.board.copy()
        boardDraw = self.board.copy()

        tmpCnts =np.zeros(self.board.shape)
        memo = np.zeros(self.board.shape)
        mask = np.zeros(self.board.shape)
        for p in self.points:
            cv.circle(mask,p, 1,255 , 1)
        cv.imshow('ret',mask)
        cv.waitKey(30)
        lastPoint = 0
        results = []
        framecnt = 0
        while np.sum(board) > 10:
            
            print(np.sum(board))
            print(framecnt)
            
            tmpMaxOff = 0
            
            tmpj= -1
            
            p1  =  points[lastPoint]
            for j in range(0,360,1):
                if lastPoint==j or memo[lastPoint][j]:
                    continue
                
                p2 = points[j]
                tmpboard = board.copy()
                

                tmpSum = np.sum(tmpboard)
                
                cv.line(tmpboard,p1,p2,0,1)
                

                tmpSum = tmpSum - np.sum(tmpboard)
                

                if tmpSum<3:
                    memo[lastPoint][j] = 1
                    continue
                if tmpMaxOff < tmpSum:
                    tmpMaxOff=tmpSum
                   
                    tmpj= j

            if tmpj != -1:
                framecnt+=1
                cv.line(board,points[lastPoint],points[tmpj],0,1)
                cv.line(boardDraw,points[lastPoint],points[tmpj],1,1)
                
                cv.line(mask,points[lastPoint],points[tmpj],255,1)
                results.append((points[lastPoint],points[tmpj]))
                memo[lastPoint][tmpj] = 1
                lastPoint=tmpj
                if framecnt%2==0:
                    tmpFrame = np.full((self.radius*2,self.radius*2,3),0,np.uint8)
                    tmpFrame[:,:,0]=mask
                    tmpFrame[:,:,1]=mask
                    tmpFrame[:,:,2]=mask
                
                    vout.write(tmpFrame)
                cv.imshow('ret',mask)
                cv.waitKey(30)
            else:
                break
        #cv.waitKey(0)
        mask[mask==255] = 1
        mask[mask==0] = 255
        mask[mask==1] = 0
        
        cv.imshow('ret',mask)
        #vout.write(mask)
        cv.imwrite('ret.jpg',mask);

        tmpFrame = np.full((self.radius*2,self.radius*2,3),0,np.uint8)
        tmpFrame[:,:,0]=mask
        tmpFrame[:,:,1]=mask
        tmpFrame[:,:,2]=mask
        for i in range(0,25*3):
            vout.write(tmpFrame)
            framecnt+=1
            print(framecnt)
        vout.release()
        cv.waitKey(20)
        cv.waitKey(0)
        print(results)
        print(framecnt)
        self.tmpCnts=tmpCnts
        self.tmpmax=tmpmax
        
        