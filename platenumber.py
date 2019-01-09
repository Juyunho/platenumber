# -*- coding: utf-8 -*-
# @Author: hyzhangyong
# @Date:   2016-06-23 16:21:54
# @Last Modified by:   hyzhangyong
# @Last Modified time: 2016-06-24 00:00:47
import sys
import cv2
import numpy as np
import os
from moviepy.editor import ImageSequenceClip

def preprocess(gray):
    # # 直方圖均衡化
    # equ = cv2.equalizeHist(gray)
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #cv2.imwrite('gaussian.jpg',gaussian)
    # 中值濾波
    median = cv2.medianBlur(gaussian, 5)
    #cv2.imwrite('median.jpg',median)
    # Sobel算子，X方向求梯度
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
    #cv2.imshow('sobel',sobel)
    #cv2.imwrite('sobel.jpg',sobel)
    # 二值化
    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary',binary)
    #cv2.imwrite('binary.jpg',binary)
    # 膨脹和腐蝕操作的核函數
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    
    # 膨脹一次，讓輪廓突出
    dilation = cv2.dilate(binary, element2, iterations = 3)
    #cv2.imshow('dilation',dilation)
    #cv2.imwrite('dilation.jpg',dilation)
    # 腐蝕一次，去掉細節
    erosion = cv2.erode(dilation, element1, iterations = 1)
    #cv2.imshow('erosion',erosion)
    #cv2.imwrite('erosion.jpg',erosion)
    # 再次膨脹，讓輪廓明顯一些
    dilation2 = cv2.dilate(erosion, element2,iterations = 3)
    #cv2.imshow('dilation2',dilation2)
    #cv2.imwrite('dilation2.jpg',dilation2)
    cv2.waitKey(0)
    return dilation

def findPlateNumberRegion(img):
    region = []
    # 查找輪廓
    binary,contours,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 篩選面積小的
    for i in range(len(contours)):
        cnt = contours[i]
        #print(i)
        
        # 計算輪廓的面積
        area = cv2.contourArea(cnt)
        
        # 面積小的都篩選掉
        if (area < 2000):
            continue
        
        # 輪廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，該矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print ("rect is: ")
        print (rect)

        # box是四个個點的座標
        box = np.int0(cv2.boxPoints(rect))
        box = np.int0(box)
        #print(box)
        #計算高和寬
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 車牌正常情況下長高比在2.7-5之間
        ratio =float(width) / float(height)
        print (ratio)
        if 2 <= ratio <= 5:
            region.append(box)
            return region
        
        if ratio > 5 or ratio < 2:
            #print("no plate")
            continue
        
        #region.append(box)

    #return region

def detect(img):
    # 轉化成灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 形態學變換的預處理
    dilation = preprocess(gray)

    # 查找車牌區域
    region = findPlateNumberRegion(dilation)
    #print(region)
    if region == None:
        print("no plate")
        return imagePath
        
    # 用綠線畫出這些找到的輪廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2) 
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)

    x1 = box[xs_sorted_index[0], 0]
    x2 = box[xs_sorted_index[3], 0]

    y1 = box[ys_sorted_index[0], 1]
    y2 = box[ys_sorted_index[3], 1]

    img_org2 = img.copy()
    img_plate = img_org2[y1:y2, x1:x2]
    #cv2.imshow('number plate', img_plate)
    platepath = "D:/code/" #mp4檔的檔名
    cv2.imwrite(platepath+"number plate"+str(i)+'.jpg', img_plate)
    

    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #cv2.imshow('img', img)

    # 帶輪廓的圖片
    cv2.imwrite('0'+str(i)+'.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    


if __name__ == '__main__':
    '''for i in range(227,230):
        name_list = [i]
        #for name in name_list:
            imagePath = (str(name_list)+".jpg")
            img = cv2.imread(imagePath)
            detect(img)
    '''
    
    videos_src_path = "D:/code"
    videos_save_path ="D:/code"

    videos = os.listdir(videos_src_path) #用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    videos = filter(lambda x: x.endswith('mp4'), videos)

    for each_video in videos:
        print(each_video)

    # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.mkdir(videos_save_path + '/' + each_video_name)
    
        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

    # get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)
    
        cap = cv2.VideoCapture(each_video_full_path)
    
        frame_count = 0
          
        if cap.isOpened():
            success, frame = cap.read()#從攝影機擷取一張影像
        else:
            success = False
            
        while (success):
            success, frame = cap.read()
        
            #print('Read a new frame: ', success)
                    
            params = []
        #params.append(cap.CV_IMWRITE_PXM_BINARY)
        #params.append(1)
            cv2.imwrite( "%d.jpg" % frame_count, frame, params)
        
            frame_count = frame_count + 1
        
            cv2.waitKey(1)
    cap.release()
    while 1:
        
        for i in range(0,frame_count-1):
            imagePath = (str(i)+".jpg")
        
            img = cv2.imread(imagePath, flags=cv2.IMREAD_COLOR)
            print(imagePath)
            detect(img)
        break
    #clip = ImageSequenceClip("D:/photo test/test/test", fps=10)
    #clip.to_videofile("D:/photo test/test/test/video.mp4", fps=10)
