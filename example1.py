#!/usr/bin/python
#-*-coding:UTF-8 -*-
import cv2
import numpy as np
import copy
from cvgame import *


right_judge = 0
left_judge = 0
up_judge = 0
down_judge = 0
judge_flag = False


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame,bgModel,learningRate): #移除背景
    fgmask = bgModel.apply(frame, learningRate=learningRate) #计算前景掩膜
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1) #使用特定的结构元素来侵蚀图像。
    res = cv2.bitwise_and(frame, frame, mask=fgmask) #使用掩膜移除静态背景
    return res

def direction_judge(deta):
    global right_judge,left_judge,up_judge,down_judge,judge_flag
    if abs(deta[0])>abs(deta[1]) and (deta[0]+abs(deta[1]))<-60:
            left_judge = 0
            up_judge = 0
            down_judge =0
            right_judge = right_judge+1
            if right_judge>10:
                print('right')
                judge_flag = False
                return 2

    elif abs(deta[0]) > abs(deta[1]) and (deta[0] - abs(deta[1])) >50:  #向左
            left_judge=left_judge+1
            right_judge=0
            up_judge = 0
            down_judge =0
            if left_judge>10:
                print('left')
                judge_flag = False
                return 1
    elif abs(deta[0]) < abs(deta[1]) and (abs(deta[0]) + deta[1])<-30:  #向下
            up_judge=0
            left_judge=0
            right_judge=0
            down_judge = down_judge+1
            if down_judge>10:
                print('down')
                judge_flag = False
                return 4

    elif abs(deta[0]) < abs(deta[1]) and (abs(deta[0]) - deta[1]) < -60: #向上
            down_judge=0
            left_judge = 0
            right_judge = 0
            up_judge = up_judge +1
            if up_judge>10:
                print('up')
                judge_flag = False
                return 3
    else:
        return 0
def main():
    # 参数
    cap_region_x_begin = 0.5  # 起点/总宽度
    cap_region_y_end = 0.8
    threshold = 60  # 二值化阈值
    blurValue = 41  # 高斯模糊参数
    bgSubThreshold = 50
    learningRate = 0

    global judge_flag
    fist_judge = 0


    # 变量
    isBgCaptured = 0  # 布尔类型, 背景是否被捕获
    triggerSwitch = False  # 如果正确，键盘模拟器将工作

    # 相机/摄像头
    camera = cv2.VideoCapture(0)   #打开电脑自带摄像头，如果参数是1会打开外接摄像头
    camera.set(10, 200)   #设置视频属性
    cv2.namedWindow('trackbar') #设置窗口名字
    cv2.resizeWindow("trackbar", 640, 200)  #重新设置窗口尺寸
    cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)
    #createTrackbar是Opencv中的API，其可在显示图像的窗口中快速创建一个滑动控件，用于手动调节阈值，具有非常直观的效果。

    matrix, currentscore, screen = init_game()
    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('threshold', 'trackbar') #返回滑动条上的位置的值（即实时更新阈值）
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
        frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)   大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
        #画矩形框  frame.shape[0]表示frame的高度    frame.shape[1]表示frame的宽度   注：opencv的像素是BGR顺序
        cv2.imshow('original', frame)   #经过双边滤波后的初始化窗口

        #主要操作
        if isBgCaptured == 1:  # isBgCaptured == 1 表示已经捕获背景
            img = removeBG(frame,bgModel,learningRate)  #移除背景
            img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # 剪切右上角矩形框区域
            #cv2.imshow('mask', img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #将移除背景后的图像转换为灰度图
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)  #加高斯模糊
            #cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)  #二值化处理
            cv2.imshow('binary', thresh)

            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #寻找轮廓   注：这里的'_'用作变量名称，_表示一个变量被指定了名称，但不打算使用。
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # 找到最大的轮廓（根据面积）
                    temp = contours[i]
                    area = cv2.contourArea(temp)  #计算轮廓区域面积
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]  #得出最大的轮廓区域
                #hull = cv2.convexHull(res)  #得出点集（组成轮廓的点）的凸包
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)   #画出最大区域轮廓
                #cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)  #画出凸包轮廓
                #cv2.imshow('no_point', drawing)
                moments = cv2.moments(res)  # 求最大区域轮廓的各阶矩
                if int(moments['m00'])!=0:
                    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cv2.circle(drawing, center, 8, (0,0,255), -1)   #画出重心

                max = 0; notice = 0
                for i in range(len(res)):
                                 temp = res[i]
                                 dist = (temp[0][0] -center[0])*(temp[0][0] -center[0]) + (temp[0][1] -center[1])*(temp[0][1] -center[1]) #计算重心到轮廓边缘的距离
                                 if dist > max:
                                     max = dist
                                     notice = i
                cv2.circle(drawing, tuple(res[notice][0]), 8, (255, 0, 0), -1)
                deta=(center[0]-res[notice][0][0],center[1]-res[notice][0][1])

                deta_abs=abs(deta[0])-abs(deta[1])
                if abs(deta_abs)<30 and judge_flag==False:
                    fist_judge = fist_judge+1
                else:
                    fist_judge=0

                if fist_judge>10:
                    judge_flag = True
                    print('fist')
                    number = 0

                if judge_flag == True:
                    number=direction_judge(deta)
                    if number != 0:
                        actionObject = GameInit.keyDownPressed(number, matrix)  # 创建各种动作类的对象
                        matrix, score = actionObject.handleData()  # 处理数据
                        currentscore += score
                        GameInit.drawSurface(screen, matrix, currentscore)
                        if matrix.min() != 0:
                            GameInit.gameOver(matrix)

                        pygame.display.update()

                cv2.imshow('output', drawing)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)


        # 输入的键盘值
        k = cv2.waitKey(10)
        if k == 27:  # 按下ESC退出
            break
        elif k == ord('b'):  # 按下'b'会捕获背景
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            #Opencv集成了BackgroundSubtractorMOG2用于动态目标检测，用到的是基于自适应混合高斯背景建模的背景减除法。
            isBgCaptured = 1
            number=0
            print('!!!Background Captured!!!')
        elif k == ord('r'):  # 按下'r'会重置背景
            bgModel = None
            isBgCaptured = 0
            print('!!!Reset BackGround!!!')


if __name__ == '__main__':
    main()