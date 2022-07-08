import shutil
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

INDEX_MAP_SIZE = 4096


def softMax(xs, temperature):
    exps = [np.exp(x * temperature) for x in xs]
    s = sum(exps)

    return [e / s for e in exps]


class Distorter:
    def __init__(self, noise_path='./temp/perlin-2.npy'):
        noise_path = [noise_path] if type(noise_path) == str else noise_path

        self.perlin_maps = [np.load(path) for path in noise_path]
        self.perlin_dimension = self.perlin_maps[0].shape[1]

        self.index_x = np.zeros((INDEX_MAP_SIZE, INDEX_MAP_SIZE), np.float32)
        self.index_y = np.zeros((INDEX_MAP_SIZE, INDEX_MAP_SIZE), np.float32)
        for y in range(INDEX_MAP_SIZE):
            for x in range(INDEX_MAP_SIZE):
                self.index_x[y, x] = x
                self.index_y[y, x] = y

    def compoundMaps(self, indices, wt):
        weights = [1] if len(self.perlin_maps) == 1 else softMax(np.random.randn(len(self.perlin_maps)), wt)

        result = np.zeros(self.perlin_maps[0][0].shape, dtype=np.float32)

        for level, maps in enumerate(self.perlin_maps):
            result += maps[indices[level]] * weights[level]

        return result

    def make_maps(self, shape, scale, intensity, wt):
        xi = [np.random.randint(len(maps)) for maps in self.perlin_maps]
        yi = [np.random.randint(len(maps)) for maps in self.perlin_maps]

        shrink = 1 / (max(shape[0], shape[1]) * scale)

        sy, sx = shape[0] * shrink, shape[1] * shrink
        biasx = np.random.random() * (1 - sx) if sx < 1 else 1 - sx
        biasy = np.random.random() * (1 - sy) if sy < 1 else 1 - sy

        nm_x = np.abs(self.index_x[:shape[0], :shape[1]] * shrink + biasx) * self.perlin_dimension
        nm_y = np.abs(self.index_y[:shape[0], :shape[1]] * shrink + biasy) * self.perlin_dimension

        noise_x = cv2.remap(self.compoundMaps(xi, wt), nm_x, nm_y, cv2.INTER_CUBIC) * intensity + self.index_x[
                                                                                                  :shape[0], :shape[1]]
        noise_y = cv2.remap(self.compoundMaps(yi, wt), nm_x, nm_y, cv2.INTER_CUBIC) * intensity + self.index_y[
                                                                                                  :shape[0], :shape[1]]

        return noise_x, noise_y

    def distort(self, source, mapx, mapy):  # source: (height, width, channel)
        ######## 针对灰度图做功能添加 ##########
        if len(source.shape) == 2:  #
            return cv2.remap(source, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        ####################################
        if source.shape[2] > 4:
            channels = source.shape[2]
            result = np.zeros((mapx.shape[0], mapx.shape[1], channels), dtype=source.dtype)
            for c in range(0, channels, 4):
                src = source[:, :, c:min(c + 4, channels)]
                result[:, :, c:min(c + 4, channels)] = cv2.remap(src, mapx, mapy, cv2.INTER_CUBIC,
                                                                 borderMode=cv2.BORDER_REPLICATE).reshape(
                    (mapx.shape[0], mapx.shape[1], min(4, channels - c)))

            return result

        return cv2.remap(source, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def do(self, img, label_img):
        """
        :param img:         原始图像
        :param label_img:   标签图像
        :return:            扭曲变换后的原始图像， 扭曲变换后的标签图像
        """
        assert img.shape[:2] == label_img.shape[:2]
        #########################
        arg_dict = {
            'scale': 2,
            'scale_sigma': 0.4,
            'intensity': 0.5,
            'intensity_sigma': 1.0, # 扭曲系数
            'noise_weights_sigma': 1
        }
        #############################

        #########################为了防止原图变换之后图像边缘的信息丢失，在原图上加白边，同时标签数据也要跟着改变#################################
        # top, bottom, left, right = 20, 20, 10, 10
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                             value=(255, 255, 255))  # 给图像加白边
        # label_img = cv2.copyMakeBorder(label_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                                value=255)  # 给图像加黑边

        ###########################################################################################################################

        scale = arg_dict['scale'] * math.exp(np.random.randn() * arg_dict['scale_sigma'])  # shrink　
        intensity = arg_dict['intensity'] * math.exp(np.random.randn() * arg_dict['intensity_sigma'])

        nx, ny = self.make_maps(img.shape, scale, intensity, arg_dict['noise_weights_sigma'])
        dealed_img = self.distort(source=img, mapx=nx, mapy=ny)
        dealed_label_img = self.distort(source=label_img, mapx=nx, mapy=ny)

        return dealed_img, dealed_label_img


if __name__ == '__main__':
    import os

    raw_img_path = r'..\..\DBnet_datasets\train_images'
    label_img_path = r'.\color_label'

    noise_config_path = r'.\temp'
    noise_list = []
    for i in os.listdir(noise_config_path):
        noise_list.append(os.path.join(noise_config_path, i))

    distorter = Distorter(noise_path=noise_list)

    arg_dict = {
        'scale': 2,
        'scale_sigma': 0.4,
        'intensity': 0.5,
        'intensity_sigma': 1.5,  # 扭曲系数
        'noise_weights_sigma': 1
    }

    for i in os.listdir(label_img_path):
        print(f'i:{i}')
        _, extend_name = os.path.splitext(i)
        if extend_name in ['.jpg', '.png']:
            scale = arg_dict['scale'] * math.exp(np.random.randn() * arg_dict['scale_sigma'])  # shrink　
            intensity = arg_dict['intensity'] * math.exp(np.random.randn() * arg_dict['intensity_sigma'])

            # scale = arg_dict['scale'] * math.exp(0.5* arg_dict['scale_sigma'])  # shrink　
            # intensity = arg_dict['intensity'] * math.exp(0.5 * arg_dict['intensity_sigma'])

            detail_label_img_path = os.path.join(label_img_path, i)
            detail_train_img_path = os.path.join(raw_img_path, i)

            raw_img = cv2.imread(detail_train_img_path)
            label_img = cv2.imread(detail_label_img_path)

            #########################为了防止原图变换之后图像边缘的信息丢失，在原图上加白边，同时标签数据也要跟着改变#################################
            top, bottom, left, right = 20, 20, 10, 10
            cv_img = cv2.copyMakeBorder(raw_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))  # 给图像加白边
            label_img = cv2.copyMakeBorder(label_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255))  # 给图像加白边

            ###########################################################################################################################
            nx, ny = distorter.make_maps(label_img.shape, scale, intensity, arg_dict['noise_weights_sigma'])
            dealed_raw_img = distorter.distort(source=raw_img, mapx=nx, mapy=ny)
            dealed_label_img = distorter.distort(source=label_img, mapx=nx, mapy=ny)

            detail_niuqu_label_path = os.path.join(save_niuqu_label_path, i)
            cv2.imwrite(detail_niuqu_label_path, dealed_label_img)

            detail_niuqu_train_path = os.path.join(save_niuqu_train_img_path, i)
            cv2.imwrite(detail_niuqu_train_path, dealed_raw_img)

            # _, ax = plt.subplots(1,3,figsize=(12,20))
            # ax[0].imshow(raw_img)
            # ax[1].imshow(dealed_raw_img)
            # ax[2].imshow(dealed_label_img)
            # plt.show()
