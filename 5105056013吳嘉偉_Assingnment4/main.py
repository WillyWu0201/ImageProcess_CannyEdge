import cv2
import numpy as np
import math

# 讀入照片
def readPhoto(name):
    return cv2.imread(name)


# 儲存照片
def savePhoto(name, im):
    cv2.imwrite(name + '.png', im)


# 取得並儲存灰階照片
def readGrayImage(name):
    grayImage = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    savePhoto('gray_image', grayImage)
    return grayImage


# 使用Canny Edge找出邊緣
def getCannyEdge(image):
    img = cv2.GaussianBlur(image, (3, 3), 0)
    cannyEdgeImage = cv2.Canny(img, 50, 150, apertureSize = 3)
    savePhoto('canny_image', cannyEdgeImage)
    return cannyEdgeImage


# Hough Transform公式
def houghTransform(x, y, angle):
    return x * math.cos(angle) + y * math.sin(angle)


# 取得Hough Transform圖片
def getHoughTransformImage(image, partNumber = 1000):
    height = image.shape[0]
    width = image.shape[1]
    minDistance = 0
    maxDistance = 0
    output_pixels = []

    for y in range(height):
        output_pixels.append([])
        for x in range(width):
            if image[y][x] != 0:
                lineDistance = []
                for count in range(partNumber):
                    angle = math.pi * (2 * count / partNumber - 1)
                    distance = houghTransform(x, y, angle)
                    lineDistance.append(distance)
                    if distance < minDistance:
                        minDistance = distance
                    if distance > maxDistance:
                        maxDistance = distance
                output_pixels[y].append(lineDistance)
            else:
                output_pixels[y].append([])

    height = round(maxDistance - minDistance)
    transData = np.zeros((height, partNumber), np.uint8)
    maxCount = 0

    for i in range(len(output_pixels)):
        row = output_pixels[i]
        for j in range(len(row)):
            line = row[j]
            for k in range(len(line)):
                p = int(line[k] - minDistance)
                transData[p][k] += 1
                if (transData[p][k] > maxCount):
                    maxCount = transData[p][k]

    newImage = np.zeros((height, partNumber, 3), np.uint8)

    for y in range(height):
        for x in range(partNumber):
            newImage[y][x] = int(transData[y][x] * 255 / maxCount)

    savePhoto('hough_transform_image', newImage)


# 畫出最長的直線
def drawLongestLine(cannyImage):
    cdst = cv2.cvtColor(cannyImage, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(cannyImage, 1, np.pi / 180, 50, None, 50, 10)
    if lines is not None:
        for i in range(0, 1):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    savePhoto('finalImage', cdst)



grayImage = readGrayImage('bridge.jpg')
cannyImage = getCannyEdge(grayImage)
getHoughTransformImage(cannyImage)
drawLongestLine(cannyImage)