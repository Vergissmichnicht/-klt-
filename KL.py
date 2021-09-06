# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob

img_num = 3  # 在此修改图片张数


class EigenFace(object):
    def __init__(self):
        self.self = self

    # 数据处理，获取训练用图片信息，同时构建数据矩阵
    def load_img(self, fileName):
        # 载入图像，灰度化处理
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        return img

    # 从文件夹中获取图片,生成图像样本矩阵
    def create_images_mat(self, dirName):
        # 查看路径下所有文件
        TrainFiles = os.listdir(dirName)

        # 计算有几个文件（图片命名都是以 序号.jpg方式）
        Train_Number = len(TrainFiles)
        X = []
        for i in range(1, Train_Number + 1):
            image = self.load_img(dirName + '/' + str(i) + '.jpg')
            image = cv2.resize(image, (500, 500))

            # 转为1-Dj矩阵
            image = image.reshape(image.size, 1)
            X.append(image)
        dataMat = np.array(X)

        dataMat = dataMat.reshape(dataMat.shape[0], dataMat.shape[1])
        return np.mat(dataMat).T

    # KL变换算法
    def PCA(self, dataMat):
        # 1.获取样本总体均值向量
        avgMat = dataMat.mean(axis=1)

        # 2.求自相关矩阵
        diffMat = dataMat - avgMat  # 获得差矩阵，转置后的行数（目前的列数）为训练的图像个数
        # print(diffMat.shape[0], diffMat.shape[1])

        covMat = np.mat(1 / img_num * diffMat.T * diffMat)
        # print(covMat.shape[0], covMat.shape[1])

        # 3.求AT*A的特征值和特征向量
        eigVals, eigVects = np.linalg.eig(covMat)

        # 4.取特征向量
        eig = []
        for i in range(covMat.shape[0]):
            # 取前20个特征向量
            eig.append(eigVects[:, i])

        eig = np.mat(np.reshape(np.array(eig), (-1, len(eig))))

        # 5.计算A *AT的特征向量
        eigenFaceMat = (diffMat * eig).T
        # print(eigenFaceMat.shape[0], eigenFaceMat.shape[1])

        # 6.将训练结束的低维矩阵转换为高维矩阵用于显示
        data = np.zeros((500, 500))
        for i in range(0, img_num):
            data = data + np.mat(eigenFaceMat[i].reshape((500, 500)))

        return data


if __name__ == '__main__':
    eigenface = EigenFace()
    faceMat = np.mat(eigenface.create_images_mat('D:/face'))
    faceMatPCA = eigenface.PCA(faceMat)
    faceMatPCA = cv2.resize(np.float64(faceMatPCA), (500, 500))
    #cv2.imwrite("D:/face/result.jpg", faceMatPCA)
    cv2.imshow('Test Result', faceMatPCA)
    cv2.waitKey(0)