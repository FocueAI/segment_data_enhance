import os.path
import cv2
from glob import glob
import numpy as np
import random
from textureSet import TextureSet

class My_mixup:

    def __init__(self, bj_img_path=r'E:\datasets\coco_datasets\coco_data'):
        '''
        :param bj_img_path:  背景图的素材地址
        '''
        self.textures = TextureSet(bj_img_path)
        self.iter = self.textures.makeIterator(frame_size = (512, 512), frame_count = 6, mono=False)

    def listadd(self,list1,list2):
        re_list = []
        for i in zip(list1,list2):
            re_list.append(i[0]+i[1])
        return re_list

    def do(self, cv_img=None, label_img=None):
        """
        :param cv_img:     原图
        :param label_img:  原图对应的标签图
        :return:  mixup后的原图， mixup后的标签图
        """
        gen_img_h, gen_img_w = cv_img.shape[0], cv_img.shape[1]
        # gen_img_h, gen_img_w = 1000, 800
        # print('welcome do function............')
        bj_img = self.iter.get((gen_img_h, gen_img_w), scale_range=(0.5, 2), blur_range=(0, 1))
        if bj_img.shape != cv_img.shape:
            bj_img = cv2.resize(bj_img,(gen_img_w,gen_img_h))
            print(f'bj_img.shape:{bj_img.shape},cv_img.shape:{cv_img.shape}')
        ########### 在该区域做常规的背景融合 以及 使用背景做 padding 操作#############
        # step1 图像融合
        # bj_weight =  random.uniform(0.1, 0.2)
        bj_weight = random.uniform(0.2, 0.6)
        print(f'-----bj_weight:{bj_weight}')
        dst_img = cv2.addWeighted(cv_img, 1-bj_weight, bj_img, bj_weight, 0)
        # step2 给图像4个边的padding
        bj_img_w, bj_img_h = int(gen_img_w*1.2), int(gen_img_h*1.2)

        bj_img_resize = cv2.resize(bj_img,(bj_img_w, bj_img_h))
        bj_img_resize_h, bj_img_resize_w = bj_img_resize.shape[0], bj_img_resize.shape[1]

        paste_x = random.randint(0,  bj_img_resize_w-gen_img_w)
        paste_y = random.randint(0, bj_img_resize_h - gen_img_h)

        bj_img_resize[paste_y:paste_y + gen_img_h, paste_x:paste_x + gen_img_w, :] = dst_img

        bj_label_img = np.zeros(shape=(bj_img_h,bj_img_w))
        bj_label_img[paste_y:paste_y + gen_img_h, paste_x:paste_x + gen_img_w] = label_img

        return bj_img_resize, bj_label_img




if __name__ == '__main__':
    def find_points_coord(list_):
        # ['1','2','3','4','5','6'] --> [(1,2),(3,4),(5,6)]
        # list_ 的长度 一定是偶数
        new_list = []
        for i in range(len(list_) // 2):
            new_list.append((int(float(list_[2 * i])), int(float(list_[2 * i + 1]))))
        return new_list

    img_path = r'./train_images'  # 图片的路径
    label_path = r'./train_gts'  # 标签的路径
    for j in os.listdir(img_path):
        file_name, extend_name = os.path.splitext(j)
        detail_img_path = os.path.join(img_path,j)
        detail_label_path = os.path.join(label_path,file_name + '.txt')

        img = cv2.imread(detail_img_path)


        with open(detail_label_path, 'r', encoding='utf-8') as reader:
            contents = reader.readlines()
        mix = My_mixup()
        deal_img, deal_label = mix.do(img, contents)
        for i in deal_label:
            tmp = i.split(',')
            cls = tmp[-1]
            points_list = find_points_coord(tmp[:-1])
            for points in zip(points_list[1:],points_list[:-1]):
                point1, point2 = tuple(list(points[0])), tuple(list(points[1]))
                cv2.line(deal_img,point1,point2,(0,255,0),2)
            cv2.line(deal_img, points_list[0], points_list[-1], (0, 255, 0), 2)


        cv2.imshow('1',deal_img)
        cv2.waitKey()






