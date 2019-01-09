# coding=utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np;
import os
import hashlib

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "I": 49,
         "J": 50, "K": 51, "L": 52, "M": 53, "N": 54, "O": 55, "P": 56, "Q": 57, "R": 58, "S": 59, "T": 60, "U": 61, "V": 62,
         "W": 63, "X": 64, "Y": 65, "Z": 66};


def md5(src):
    myMd5 = hashlib.md5()
    myMd5.update(src)
    myMd5_Digest = myMd5.hexdigest()
    return myMd5_Digest


def r(factor):
    return int(np.random.random() * factor);


def debugshow(img):
    win_name = md5(img)[:5]
    cv2.imshow("img", img)
    cv2.waitKey(0)


def setPadding(img, padding):
    return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT);


def rotRandrom(img, factor, size):
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [0, shape[0] - r(factor)], [shape[1] - r(factor), 0],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;


def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)

    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50];
    adder = cv2.resize(adder, (50, 50));
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)

    return img


def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));


def Addunblance(img):
    for x in range(img.shape[0]):
        img[x, :] -= -10 + r(10)


def Addshade(img, size, i):
    rows = r(size)
    cols = r(size);

    if (i == 0):
        shade = np.zeros((rows, 50, 3), dtype=np.uint8);
        img[50 - rows:50, 0:50] = shade;
    if (i == 1):
        shade = np.zeros((rows, 50, 3), dtype=np.uint8);
        img[0:rows, 0:50] = shade;
    if (i == 2):
        shade = np.zeros((50, cols, 3), dtype=np.uint8);
        img[0:50, 0:cols] = shade;
    if (i == 3):
        shade = np.zeros((50, cols, 3), dtype=np.uint8);
        img[0:50, 50 - cols:50] = shade;
    return img;


def thes(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);

    flags, thes = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU);
    thes = cv2.cvtColor(thes, cv2.COLOR_GRAY2BGR);

    return thes;


class genSamples:
    chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ];

    def __init__(self, chineseFontpath, EnglishFontPath, smuImgPath):
        self.fontC = ImageFont.truetype(chineseFontpath, 43, 0);
        self.fontE = ImageFont.truetype(EnglishFontPath, 66, 0);
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.smu = cv2.imread(smuImgPath);

    def GenCh(self, f, val):
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((3, 0), val, (0, 0, 0), font=f)

        A = np.array(img)
        A = A[10:50, 0:50]
        A = cv2.resize(A, (50, 50))

        return A

    def GenEng(self, f, val):
        img = Image.new("RGB", (50, 55), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((13, 0), val, (0, 0, 0), font=f)

        A = np.array(img)
        A = A[10:55, 0:50]
        A = cv2.resize(A, (50, 50))
        return A

    def randomWindows(self, img, scale):
        size = int(img.shape[0])
        zoom = int((size * scale))
        windows_size = int(size - zoom)
        rand_range = zoom;
        rand_x = int(r(rand_range))
        rand_y = int(r(rand_range))

        new_img = img[rand_x:rand_x + windows_size, rand_y:rand_y + windows_size];
        new_img = cv2.resize(new_img, (50, 50));
        return new_img;

    def genImage(self,c_batch_num,id, tranformFactor=7, shadeSize=10, shadeFilter=[], smuFilter=[], blur=3,rotFilter = [],blurFilter_level1=[],
                 blurFilter_level2=[], size=100):

        if (id > 30):
            img = self.GenEng(self.fontE, self.chars[id]);
            img = cv2.bitwise_not(img)
        else:
            img = self.GenCh(self.fontC, self.chars[id]);
            img = cv2.bitwise_not(img)
            border = r(tranformFactor);
            side = border*2
            img = cv2.resize(img, (40-14, 50))
            img = cv2.copyMakeBorder(img, 0, 0, 7, 7, cv2.BORDER_CONSTANT);
        if id not in rotFilter and (c_batch_num<300 or (c_batch_num>600  and c_batch_num>750 )):
            img = rotRandrom(img, tranformFactor, (img.shape[1], img.shape[0]));
        if id not in shadeFilter and c_batch_num<600:
            img = self.randomWindows(img, 0.100);
        # 添加遮罩
        if id not in smuFilter and c_batch_num<900:
            img = AddSmudginess(img, self.smu);
        # 添加污迹
        if id not in blurFilter_level2 and c_batch_num<1200:
            if id in blurFilter_level1:
                img = AddGauss(img, r(blur - 1));
            else:
                img = AddGauss(img, r(blur) + 1);
        # 添加模糊
        img = thes(img)
        img = setPadding(img, 0)
        # 阈值
        img = cv2.resize(img, (size, size))
        #debugshow(img)
        # 20 *20 图像
        return img;

    def genBatch(self, batchSize, charRange, outputPath, tranformFactor=10, shadeSize=5, shadeFilter=[], smuFilter=[],
                 blur=0,rotFilter = [], blurFilter_level1=[], blurFilter_level2=[], size=299):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in range(batchSize):
            print("generate Batch:",i)
            for j in charRange:
                img = self.genImage(i,j, tranformFactor, shadeSize, shadeFilter, smuFilter, blur,rotFilter, blurFilter_level1,
                                    blurFilter_level2, size);
                dir = outputPath + "/" + str(j).zfill(2)
                if (not os.path.exists(dir)):
                    os.mkdir(dir)
                cv2.imwrite(dir + "/" + str(i).zfill(2) + ".jpg", img);


if __name__ == '__main__':
    shadefilter = [index["T"], index["L"], index["7"], index["0"], index["D"], index["Q"] + index["2"],index["Z"]]+list(range(31));

    print(shadefilter);
    Generator = genSamples("./font/msgothic.ttc", "./font/platechar.ttf", "./images/smu3.jpg");
    Generator.genBatch(100, list(range(67)), "./samples", blur=1,
                       shadeFilter=shadefilter);
