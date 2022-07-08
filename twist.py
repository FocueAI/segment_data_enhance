import os, cv2, math
import numpy as np
from perlin import Distorter
from warpper import My_warpPerspective
from mixup import My_mixup
import matplotlib.pyplot as plt
import  enhance_tools as tool
from enhance_tools import Cv2Pil,Pil2Cv2

class Twist:

    def __init__(self, noise_config_path=r'.\temp',bj_imgs_path=r'E:\datasets\coco_datasets\coco_data'):

        # self.arg_dict = {
        #     'scale': 2,
        #     'scale_sigma': 0.4,
        #     'intensity': 0.5,
        #     'intensity_sigma': 1.0,  # 扭曲系数
        #     'noise_weights_sigma': 1
        # }

        self.noise_list = []
        for i in os.listdir(noise_config_path):
            self.noise_list.append(os.path.join(noise_config_path, i))

        self.distorter = Distorter(noise_path=self.noise_list)  # perlin噪声增强
        self.warpper = My_warpPerspective()  # 透射变换增强
        self.mixup = My_mixup(bj_img_path=bj_imgs_path)


    def transformer(self, train_img, label_img):
        '''
        :param train_img:  训练图片
        :param label_img:  标签图片
        :return:  经过 perlin噪声 扭曲的 训练图片和标签图片
        '''
        #########################为了防止原图变换之后图像边缘的信息丢失，在原图上加白边，同时标签数据也要跟着改变#################################
        top, bottom, left, right = 30, 30, 30, 30
        cv_img = cv2.copyMakeBorder(train_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))  # 给图像加白边
        label_img = cv2.copyMakeBorder(label_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=(255, 255, 255))  # 给图像加白边

        ###########################################################################################################################
        scale = self.arg_dict['scale'] * math.exp(np.random.randn() * self.arg_dict['scale_sigma'])  # shrink　
        intensity = self.arg_dict['intensity'] * math.exp(np.random.randn() * self.arg_dict['intensity_sigma'])
        nx, ny = self.distorter.make_maps(label_img.shape, scale, intensity, self.arg_dict['noise_weights_sigma'])
        dealed_raw_img = self.distorter.distort(source=cv_img, mapx=nx, mapy=ny)
        dealed_label_img = self.distorter.distort(source=label_img, mapx=nx, mapy=ny)

        return dealed_raw_img, dealed_label_img

    def do(self, cv_img, labels_img):
        ################ 与背景merge ############# begin
        if np.random.uniform() > 0:
            cv_img, labels_img = self.mixup.do(cv_img, labels_img)
        ######################################## end

        ################ 常规的数据增强 (主要针对原图操作、标签图无需操作)############# begin
        # 1. 高斯模糊
        if np.random.uniform() > 0:
            cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)
        # 2. 椒盐噪声
        if np.random.uniform() > 0:
            enhance_factor = np.random.uniform(0.001, 0.01)
            cv_img = tool.SaltAndPepper(cv_img, enhance_factor)
        # 3. 图像旋转
        if np.random.uniform() > 0:
            cv_img = Cv2Pil(cv_img)
            labels_img = Cv2Pil(labels_img,is_gray=True)

            rotate_angle = np.random.randint(-5,5)
            cv_img = cv_img.rotate(rotate_angle)
            labels_img = labels_img.rotate(rotate_angle)

            cv_img = Pil2Cv2(cv_img)
            labels_img = Pil2Cv2(labels_img,is_gray=True)

            print(f'图像旋转来了，，，，，，')
        ######################################## end

        ################ 透射变换 ################# begin
        if np.random.uniform() > 0:
            cv_img, labels_img = self.warpper.do(cv_img, labels_img)
        ######################################### end

        ################ perlin噪声增强-让图像扭曲 ################# begin
        if np.random.uniform() > 0:
            try:
                cv_img, labels_img = self.distorter.do(cv_img, labels_img)
            except Exception as e:
                cv_img, labels_img = cv_img, labels_img
                print(f'find an err:{e}')
        ######################################################## end
        print(f'cv_img-shape:{cv_img.shape}')
        print(f'labels_img-shape:{labels_img.shape}')

        return cv_img, labels_img





if __name__ == '__main__':

    # 图片x的路径
    img_path = r'D:\my_projects\find\seg_datasets\manmake-like-public\VOCdevkit\VOC2007\JPEGImages'
    # 标签图片y的路径
    label_path = r'D:\my_projects\find\seg_datasets\manmake-like-public\VOCdevkit\VOC2007\SegmentationClass'

    for j in os.listdir(img_path):
        if j.endswith('.jpg'):
            file_name, extend_name = os.path.splitext(j)
            detail_img_path = os.path.join(img_path, j)
            detail_label_path = os.path.join(label_path, file_name + '.png')

            img_x = cv2.imread(detail_img_path)
            label_y = cv2.imread(detail_label_path,0)
            twist = Twist()

            img_x_enhance, label_y_enhance = twist.do(img_x, label_y)

            cv2.imshow('img_x_enhance',img_x_enhance)
            cv2.imshow('label_y_enhance',label_y_enhance)
            cv2.waitKey()
            # ax1 = plt.subplot(1, 2, 1)
            # ax2 = plt.subplot(1, 2, 2)
            #
            # im1 = ax1.imshow(img_x_enhance,aspect='auto')
            # im2 = ax2.imshow(label_y_enhance,cmap=plt.cm.gray,vmin=0,vmax=255,aspect='auto')
            #
            # plt.show()

