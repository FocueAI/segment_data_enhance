import os
import random
import cv2
import numpy as np


def find_points_coord(list_):
    # ['1','2','3','4','5','6'] --> [(1,2),(3,4),(5,6)]
    # list_ 的长度 一定是偶数
    new_list = []
    for i in range(len(list_) // 2):
        new_list.append((int(float(list_[2 * i])), int(float(list_[2 * i + 1]))))
    return new_list

class My_warpPerspective:
    def __init__(self,padding_size=(30,30,30,30)):
        self.top = padding_size[0]
        self.bottom = padding_size[1]
        self.left = padding_size[2]
        self.right = padding_size[3]

    def point_map(self,M,point):
        '''
        :param M:      仿射变换矩阵
        :param point:  原图上的 一点 的坐标
        :return:       仿射变换后 相应点 对应的坐标
        '''
        x, y = point
        point_1 = np.array([[x+self.left, y+self.top]], dtype='float32')
        point_1 = np.array([point_1])
        dst = cv2.perspectiveTransform(point_1, M)
        x = int(dst[0][0][0])
        y = int(dst[0][0][1])
        return (x,y)

    def img_merge_bj(self,bj_path_list):
        pass


    def do(self,cv_img, label_img):
        '''
        :param cv_img:           要进行透射变换的 图像
        :param label_contents:   ['x0,y0,x1,y1,x2,y2,label','',.......]
        :return:  扭曲后的cv_img, 扭曲后的 label_contents
        '''
        #########################为了防止原图变换之后图像边缘的信息丢失，在原图上加白边，同时标签数据也要跟着改变#################################

        cv_img = cv2.copyMakeBorder(cv_img, self.top, self.bottom, self.left, self.right, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))  # 给图像加白边

        label_img = cv2.copyMakeBorder(label_img, self.top, self.bottom, self.left, self.right, cv2.BORDER_CONSTANT,
                                    value=0)  # 给标签图像加 黑边

        assert cv_img.shape[:2] == label_img.shape[:2]
        ##########################################################################################################################
        img_h, img_w, _ = cv_img.shape
        pts1 = np.float32([[0, 0],  # 原图透射前的 4个定点坐标
                           [img_w, 0],
                           [img_w, img_h],
                           [0, img_h]
                           ])
        # random_border_x = random.randint(1, max(2 ,min(img_w//20, 10)) )
        # random_border_y = random.randint(1, max(2 ,min(img_h//8, 5)) )

        random_border_x = random.randint(20,50) # max(2 ,min(img_w//20, 3)) )
        random_border_y = random.randint(20,50) # max(2 ,min(img_h//8, 50)) )
        ######################################################
        # 1  平行四边形 1
        # random_shift_x = random.randint(0, random_border)
        pts2_1 = np.float32([ [random_border_x, random_border_y],
                              [img_w +random_border_x, random_border_y],
                              [img_w, img_h],
                              [0 ,img_h]
                              ])
        # 2  平行四边形 2
        pts2_2 = np.float32([ [-random_border_x, random_border_y],
                              [img_w -random_border_x, random_border_y],
                              [img_w, img_h],
                              [0 ,img_h]
                              ])
        # # 3  梯形 1
        pts2_3 = np.float32([ [random_border_x, random_border_y],
                              [img_w -random_border_x, random_border_y],
                              [img_w, img_h],
                              [0 ,img_h]
                              ])
        # # 3  梯形 2
        pts2_4 = np.float32([ [0, 0],
                              [img_w, 0],
                              [img_w -random_border_x, img_h -random_border_y],
                              [random_border_x ,img_h -random_border_y]
                              ])

        # pst2_list = [pts2_1,pts2_2,pts2_3,pts2_4]
        pst2_list = [pts2_1, pts2_2, pts2_3, pts2_4]
        pts2 = pst2_list[random.randint(0 ,len(pst2_list ) -1)]

        M = cv2.getPerspectiveTransform(pts1, pts2)


        transform_img = cv2.warpPerspective(cv_img, M, (int(img_w), int(img_h)), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))

        transform_label_img = cv2.warpPerspective(label_img, M, (int(img_w), int(img_h)), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        ## 其中 x0，y0, x1, y1  为原图像上的点，  x0_，y0_, x1_, y1_  为透射变换之后的对应点坐标
        transform_img_h, transform_img_w = transform_img.shape[0], transform_img.shape[1]
        transform_label_img_h, transform_label_img_w = transform_label_img.shape[0], transform_label_img.shape[1]

        assert transform_img.shape[0:2] == transform_label_img.shape[0:2]

        return transform_img, transform_label_img

if __name__ == '__main__':
    img_path = r'./train_images'
    label_path = r'./train_gts'
    warpPerspective = My_warpPerspective()
    for i in os.listdir(img_path):
        file_name, _ = os.path.splitext(i)
        detail_img_path = os.path.join(img_path, i)
        detail_label_path = os.path.join(label_path, file_name+'.txt')

        cv_img = cv2.imread(detail_img_path)
        with open(detail_label_path,'r',encoding='utf-8') as reader:
            label_contents = reader.readlines()

        deal_img, deal_label = warpPerspective.do(cv_img, label_contents)
        ########## 给处理过的图片画上标签，以此检查标签是否正确 ##########
        for j in deal_label:
            box_info = j.split(',')
            cv2.line(deal_img,( int(float(box_info[0])),int(float(box_info[1])) ),( int(float(box_info[2])),int(float(box_info[3])) ),(0,0,255),1)
            cv2.line(deal_img, (int(float(box_info[4])), int(float(box_info[5]))),(int(float(box_info[2])), int(float(box_info[3]))), (0, 0, 255), 1)
            cv2.line(deal_img, (int(float(box_info[4])), int(float(box_info[5]))),
                     (int(float(box_info[6])), int(float(box_info[7]))), (0, 0, 255), 1)
            cv2.line(deal_img, (int(float(box_info[0])), int(float(box_info[1]))),
                     (int(float(box_info[6])), int(float(box_info[7]))), (0, 0, 255), 1)
        ####################
        print('=============图像名字:%s=====图像box数量:%s========'%(i,len(deal_label)))

        print(f'deal_label:{deal_label}')
        cv2.imshow(f'{i}',deal_img)
        cv2.waitKey()

