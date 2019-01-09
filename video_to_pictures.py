#Convert the video to images and store to video output
#! encoding:UTF-8

import os
import cv2

videos_src_path = 'D:/code/picture/video10'
videos_save_path = 'D:/code/picture/video10'

videos = os.listdir(videos_src_path) #用於返回指定的文件夾包含的文件或文件夾的名字的列表。
videos = filter(lambda x: x.endswith('avi'), videos)

for each_video in videos:
    print(each_video)

    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')
    os.mkdir(videos_save_path + '/' + each_video_name)
    
    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

    # get the full path of each video, which will open the video tp extract frames
    each_video_full_path = os.path.join(videos_src_path, each_video)
    
    cap = cv2.VideoCapture(each_video_full_path)
    frame_count = 1

    if cap.isOpened():
        success, frame = cap.read()#從攝影機擷取一張影像
    else:
        success = False
            
    while (success):
        success, frame = cap.read()

        # 將圖片轉為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print('Read a new frame: ', success)
        
        params = []
        #params.append(cap.CV_IMWRITE_PXM_BINARY)
        #params.append(1)
        cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, gray, params)
        
        frame_count = frame_count + 1
        
        cv2.waitKey(1)

cap.release()
