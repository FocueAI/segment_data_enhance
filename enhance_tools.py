import numpy as np
import cv2
import os
from PIL import Image
import random

# 为了防止图像经过变换之后出现锯齿化之后 begin
def cover_resolution(cv_img,scale_factor):
    """
       知识点：
           增加分辨率的方式
           INTER_NEAREST（最近邻插值）
           INTER_CUBIC  (三次样条插值)
           INTER_LINEAR(线性插值)
           INTER_AREA  (区域插值)
       输入：
           cv_img: 原图像
           scale_factor : 缩放因子  大于0 的数,  大于1 表示放大， 小于1 表示缩小
       输出：
           放缩后的图像

    """
    before_h, before_w, _ = cv_img.shape
    return cv2.resize(cv_img,int(before_w*scale_factor),int(before_h*scale_factor),interpolation=cv2.INTER_CUBIC)



################################ end

def Cv2Pil(cv_img,is_gray=False):
    if not is_gray:
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(cv_img)
    return pil_img

def Pil2Cv2(pil_img,is_gray=False):
    if not is_gray:
        cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    else:
        cv_img = np.asarray(pil_img)
    return cv_img

def Random_border(img, random_padding_border_lower, random_padding_border_upper):
    """
    :param img:                       numpy类型的数据
    :param random_padding_border:     随机
    :return:
    """
    assert random_padding_border_upper > random_padding_border_lower


    top_padding = np.random.randint(random_padding_border_lower, random_padding_border_upper)
    bottom_padding = np.random.randint(random_padding_border_lower, random_padding_border_upper)
    left_padding = np.random.randint(random_padding_border_lower, random_padding_border_upper)
    right_padding = np.random.randint(random_padding_border_lower, random_padding_border_upper)
    #                        原图      上             下            左            右               固定值-常量          常量的值
    img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    return img

def SaltAndPepper(image ,percetage):
    '''
    :param image:         opencv格式的图片
    :param percetage:     percetage 噪声占整个图片像素数量的比例
    :return:              含有 椒盐噪声的图片
    '''
    SP_NoiseImg =image.copy()
    SP_NoiseNum =int(percetage *image.shape[0] *image.shape[1])
    for i in range(SP_NoiseNum):
        randR =np.random.randint(0 ,image.shape[0 ] -1)
        randG =np.random.randint(0 ,image.shape[1 ] -1)
        randB =np.random.randint(0 ,3)
        if np.random.randint(0 ,1)==0:
            SP_NoiseImg[randR ,randG ,randB ] =0
        else:
            SP_NoiseImg[randR ,randG ,randB ] =255
    return SP_NoiseImg

def addGaussianNoise(image ,percetage):
    '''
    :param image:         opencv格式的图片
    :param percetage:     percetage 噪声占整个图片像素数量的比例
    :return:              含有 高斯噪声的图片
    '''
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum =int(percetage *image.shape[0] *image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0 ,h)
        temp_y = np.random.randint(0 ,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def darker(image ,percetage=0.9):
    '''
    :param image:         opencv格式的图片
    :param percetage:     percetage<1 ,表示像素值变为原先的   1/percetage
    :return:              变暗后的图片
    '''
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    assert percetage<=1
    for xi in range(0 ,w):
        for xj in range(0 ,h):
            image_copy[xj ,xi ,0] = int(image[xj ,xi ,0 ] *percetage)
            image_copy[xj ,xi ,1] = int(image[xj ,xi ,1 ] *percetage)
            image_copy[xj ,xi ,2] = int(image[xj ,xi ,2 ] *percetage)
    return image_copy

def brighter(image, percetage=1.5):
    '''
    :param image:         opencv格式的图片
    :param percetage:     percetage>1 ,表示像素值变为原先的 percetage 倍
    :return:              变暗后的图片
    '''
    assert percetage>=1
    image_copy = image.copy()
    h = image.shape[0]
    w = image.shape[1]
    # get brighter
    for xi in range(0 ,w):
        for xj in range(0 ,h):
            image_copy[xj ,xi ,0] = np.clip(int(image[xj ,xi ,0 ] *percetage) ,a_max=255 ,a_min=0)
            image_copy[xj ,xi ,1] = np.clip(int(image[xj ,xi ,1 ] *percetage) ,a_max=255 ,a_min=0)
            image_copy[xj ,xi ,2] = np.clip(int(image[xj ,xi ,2 ] *percetage) ,a_max=255 ,a_min=0)
    return image_copy

def rotate(image, angle=15, scale=0.9):
    '''
    :param image:  opencv格式的图片
    :param angle:  旋转的角度
    :param scale:  图片的缩放比例
    :return:       旋转后的图片
    '''
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D(( w /2 , h /2), angle, scale)
    # rotate
    image = cv2.warpAffine(image ,M ,(w ,h))
    return image

# def img_augmentation(path, name_int):
#     img = cv2.imread(path)
#     img_flip = cv2.flip(img ,1  )  # flip
#     img_rotation = rotate(img  )  # rotation
#
#     img_noise1 = SaltAndPepper(img, 0.3)
#     img_noise2 = addGaussianNoise(img, 0.3)
#
#     img_brighter = brighter(img)
#     img_darker = darker(img)
#
#     cv2.imwrite(save_path +'%s' %str(name_int) + '.jpg', img_flip)
#     cv2.imwrite(save_path + '%s' % str(name_int + 1) + '.jpg', img_rotation)
#     cv2.imwrite(save_path + '%s' % str(name_int + 2) + '.jpg', img_noise1)
#     cv2.imwrite(save_path + '%s' % str(name_int + 3) + '.jpg', img_noise2)
#     cv2.imwrite(save_path + '%s' % str(name_int + 4) + '.jpg', img_brighter)
#     cv2.imwrite(save_path + '%s' % str(name_int + 5) + '.jpg', img_darker)
#     print('over')
# # TODO: https://blog.csdn.net/u011984148/article/details/107572526
#
# TODO： 1. 自己先实现了一版本较为粗糙的随机擦除的问题
def random_cut(image,min_factor=0.2,max_factor=0.6,replace_color=(128,128,128)):
    '''
    :param image:  待处理的图片
    :param replace_color:  None: 随机颜色， 否则，给定灰色
    :param min_factor: 擦除区域  宽 or 高的最小占比
    :param max_factor: 擦除区域  宽 or 高的最大占比
    :return:  擦出部分区域后的图片
    '''
    w = image.shape[1]
    h = image.shape[0]

    eraser_zone_width = random.randint(int(w * min_factor), int(w * max_factor) )
    eraser_zone_height = random.randint(int(h * min_factor), int(h * max_factor) )
    print('eraser_zone_width:', eraser_zone_width)
    print('eraser_zone_height:',eraser_zone_height)
    left_x = random.randint(0,w-eraser_zone_width)
    left_y = random.randint(0,h-eraser_zone_height)

    r,g,b = replace_color
    image[left_y:eraser_zone_height,left_x:eraser_zone_width,:] = [b,g,r]

    return image
# TODO： 2. 自己先实现了一版本较为粗糙的 Hide and Seek
#
def hidden_patch(image,grid_size=(16,16),hidden_rate=0.2):
    '''
    将图像分割成一个由SxS patch组成的网格。以一定的概率隐藏每个补丁(p_hide)。这允许模型了解物体的样子，而不只是学习物体的单个部分是什么样子。
    :param image:      输入的图片
    :param grid_size:  栅格的规格  eg: (8,8)-->(高,宽)   即把图片分成8*8的小patches
    :param hidden_rate: patch 隐藏的概率
    :return:    处理后的图片
    '''
    w = image.shape[1]
    h = image.shape[0]
    grid_hight_base = h//grid_size[0]
    grid_width_base = w//grid_size[1]
    for row_ in range(grid_size[0]):      # 行  ---> 对应高的部分
        for list_ in range(grid_size[1]): # 列  ---> 对应宽的部分
            if random.random() < hidden_rate :
                # if  row_%2 == 0 or list_%2 == 0:   # 可以去除该条件
                    image[int(row_*grid_hight_base):int((row_+1)*grid_hight_base), int(list_*grid_width_base):int((list_+1)*grid_width_base),: ] = [0,0,0]
    return image



def grid_mask(image, grid_size=(16, 16)):
    '''
    :param image:      输入的图片
    :param grid_size:  栅格的规格  eg: (8,8)-->(高,宽)   即把图片分成8*8的小patches
    :return: 处理后的图片
    '''
    w = image.shape[1]
    h = image.shape[0]

    grid_hight_base = h // grid_size[0]
    grid_width_base = w // grid_size[1]
    for row_ in range(grid_size[0]):  # 行  ---> 对应高的部分
        for list_ in range(grid_size[1]):  # 列  ---> 对应宽的部分
            if row_ % 2 != 0 and list_ % 2 != 0:
                image[int(row_ * grid_hight_base):int((row_ + 1) * grid_hight_base), int(list_ * grid_width_base):int((list_ + 1) * grid_width_base), :] = [0, 0, 0]
    return image


def MixUp(image):
    pass

def CutMix(image):
    pass

def Mix(img1,img2,img3,img4):
    ''' opencv 格式的图片img1,img2,img3,img4的拼接 '''
    min_offset = 0.2
    w = img1.shape[1]
    h = img1.shape[0]
    cut_x = np.random.randint(int(w * min_offset), int(w * (1 - min_offset)))
    cut_y = np.random.randint(int(h * min_offset), int(h * (1 - min_offset)))


    d1 = img1[ :(h - cut_y), 0:cut_x, :]   # 左上角

    d2 = img2[ (h - cut_y):, 0:cut_x, :]   # 左下角

    d3 = img3[ (h - cut_y):, cut_x:, :]    # 右下角

    d4 = img4[ :(h - cut_y), cut_x:, :]    # 右上角


    tmp1 = np.vstack((d1, d2))        # 垂直方向 （行顺序）
    tmp2 = np.vstack((d4, d3))        # 垂直方向 （行顺序）

    tmpx = np.hstack((tmp1, tmp2))  # 水平方向  (列顺序）
    tmpx = tmpx * 255


    return tmpx

############
def color_shake(cv2_img):
    '''
    :param cv2_img:
    :return: 颜色抖动后的图片
    '''
    image = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img = rgb_to_hsv(np.array(image) / 255.)
    img = hsv_to_rgb(img) * 255.
    # img = np.transpose(img,(2,1,0))
    return img

def rgb_to_hsv(arr):
    """
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    # make sure it is an ndarray
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_ndim = arr.ndim
    if arr.ndim == 1:
        arr = np.array(arr, ndmin=2)

    # make sure we don't have an int image
    arr = arr.astype(np.promote_types(arr.dtype, np.float32))

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    if in_ndim == 1:
        out.shape = (3,)

    return out


def hsv_to_rgb(hsv):
    """
    convert hsv values in a numpy array to rgb values
    all values assumed to be in range [0, 1]

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    # if we got passed a 1D array, try to treat as
    # a single color and reshape as needed
    in_ndim = hsv.ndim
    if in_ndim == 1:
        hsv = np.array(hsv, ndmin=2)

    # make sure we don't have an int image
    hsv = hsv.astype(np.promote_types(hsv.dtype, np.float32))

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    # `np.stack([r, g, b], axis=-1)` (numpy 1.10).
    rgb = np.concatenate([r[..., None], g[..., None], b[..., None]], -1)

    if in_ndim == 1:
        rgb.shape = (3,)

    return rgb


###########################---<弹性形变>----##########################################
# 参考文档： https://zhuanlan.zhihu.com/p/46833956
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
    # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
    # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
    # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
    # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))





def find_file_name(undealed_file):
    image_name = ''
    dealed_file = undealed_file.split('.')
    if len(dealed_file) < 2:
        raise TypeError
    if len(dealed_file) == 2:
        image_name = dealed_file[0]
    elif len(dealed_file) > 2:
        for no, i in enumerate(dealed_file):
            if no == len(dealed_file) - 1:
                pass
            else:
                image_name = image_name + i + '.'
    if image_name.endswith('.'):
        image_name = image_name[:-1]
    return image_name
######################################################

if __name__ == '__main__':

    src_path = r'./img_raw'       # 待处理的图片
    dst_path = r'./img_enhance'  # 图片增强后的存储结果
    if os.path.exists(dst_path): shutil.rmtree(dst_path)
    os.mkdir(dst_path)

    PRINT_FLAG = True
    for j in range(100):
        for i in tqdm(os.listdir(src_path)):
            if 'jpg' in i:
                file_name = find_file_name(i)
                detail_src_path = os.path.join(src_path,i)
                detail_dst_path = os.path.join(dst_path,file_name + '_enhance_%s.jpg'%(j))

                img = cv2.imread(detail_src_path)
                ###  step1: 变亮
                if random.random() > 0.5:
                    enhance_factor = random.uniform(1.2,1.5)
                    img = tool.brighter(img,enhance_factor)
                    if PRINT_FLAG: print('step1:变亮')
                ###  step2: 变暗
                if random.random() > 0.5:
                    enhance_factor = random.uniform(0.7, 0.9)
                    img = tool.darker(img, enhance_factor)
                    if PRINT_FLAG: print('step2:变暗')
                ### step3：高斯模糊
                if random.random() > 0.5:
                    enhance_factor = random.uniform(0.01, 0.04)
                    img = tool.addGaussianNoise(img, enhance_factor)
                    if PRINT_FLAG: print('step3：高斯模糊')
                ### step4： 椒盐噪声
                if random.random() > 0.5:
                    enhance_factor = random.uniform(0.01, 0.04)
                    img = tool.SaltAndPepper(img, enhance_factor)
                    if PRINT_FLAG: print('step4：椒盐噪声')
                ### step5： 随机反转（垂直 or 水平）
                if random.random()>0.5:
                    img = cv2.flip(img, random.randint(0,1))  # 0：垂直 1：水平  -1：水平+ 垂直
                    if PRINT_FLAG: print('step5： 随机反转（垂直 or 水平）')
                # step6： 随机反转（垂直 and 水平）
                if random.random()>0.5:
                    img = cv2.flip(img, -1)
                    if PRINT_FLAG: print('step6： 随机反转（垂直 and 水平）')
                ### step7： 仿射变换
                img_h, img_w, _ = img.shape
                pts1 = np.float32([[0, 0],  # 原图透射前的 4个定点坐标
                                   [img_w, 0],
                                   [img_w, img_h],
                                   [0, img_h]
                                   ])
                random_border = random.randint(5, 40)
                pts2 = np.float32([[np.random.randint(-random_border, random_border), np.random.randint(-random_border, random_border)],  # 将原图上的 4个定点 透射到目标点上 ********* 在这里需要经常的变换
                                   [img_w + np.random.randint(-random_border, random_border), np.random.randint(-random_border, random_border)],
                                   [img_w + np.random.randint(-random_border, random_border), img_h + np.random.randint(-random_border, random_border)],
                                   [np.random.randint(-random_border, random_border), img_h + np.random.randint(-random_border, random_border)]
                                   ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                if random.random() > 0.5:
                    img = cv2.warpPerspective(img, M, (int(img_w), int(img_h)))
                    if PRINT_FLAG: print('step7： 仿射变换')
                ### step8： 颜色抖动
                if random.random() > 0.5:
                    img = tool.color_shake(img)
                    if PRINT_FLAG: print('step8： 颜色抖动')
                if random.random() > 0.5:
                    img = tool.random_cut(img)
                cv2.imwrite(detail_dst_path, img)









