# from include import *
from s_utils import *
from code_decoder_manager import *
import random
import math
import os
from file_iterator import *


class AbsTransfer(object):
    def __init__(self):
        pass

    def change(self, img, label):
        """
        :param img:
        :param label:
        :return: img, label
        """
        pass


class TransferManager(object):
    def __init__(self, AbsTransfer=None):
        self.__transfer = AbsTransfer

    def set_transfer(self, AbsTransfer=None):
        self.__transfer = AbsTransfer

    def change(self, img, label):
        return self.__transfer.change(img, label)


class CropFace(AbsTransfer):
    def __init__(self, face_size=24):
        self.__face_size = face_size

    def get_random_box(self, face_box):
        """
        face_box (x1, y1, x2, y2)
        :param face_box:
        :return:
        """
        w_face = face_box[2] - face_box[0]
        h_face = face_box[3] - face_box[1]
        h_max = max(w_face, h_face)
        h_min = min(w_face, h_face)

        if h_max == h_min:
            return face_box

        n_rand = np.random.randint(0, h_max - h_min, 1)[0]

        if w_face > h_face:
            return (face_box[0], face_box[1] - n_rand, face_box[0] + h_max, face_box[1] - n_rand + h_max)

        if w_face < h_face:
            return (face_box[0] - n_rand, face_box[1], face_box[0] - n_rand + h_max, face_box[1] + h_max)

    def crop_square_box(self, attr, face_box):
        """
        :param img_box:　(width, height)
        :param face_box: (x1, y1, x2, y2)
        :return:
        """
        width = attr[0]
        height = attr[1]
        w_face = face_box[2] - face_box[0]
        h_face = face_box[3] - face_box[1]
        square = max(w_face, h_face)

        std_iou = iou(np.array([0, 0, square, square], np.int32), np.array([[0, 0, width, height]], np.int32))

        for i in range(10):
            quare_box = self.get_random_box(face_box)

            q_iou = iou(np.array(quare_box, np.int32), np.array([[0, 0, width, height]], np.int32))

            """
            all must be >0
            """
            if q_iou == std_iou and min(quare_box) >= 0:
                return quare_box

        return False

    def change(self, img, label):
        check_num = lambda item: item['num'] < 1
        check_box_size = lambda box: min(r_box[2] - r_box[0], r_box[3] - r_box[1]) <= self.__face_size
        attr = (label['width'], label['height'])

        """
        must have at least one anno
        """
        if check_num(label):
            return False

        r_box = random.sample(label['annos'], 1)[0]

        if check_box_size(r_box):
            return False

        s_box = self.crop_square_box(attr, r_box)

        if s_box is False:
            return False

        scale = float(self.__face_size / (s_box[2] - s_box[0]))

        resize_img = img_resize(img, attr, scale)

        r_box = list(map(math.ceil, [i*scale for i in s_box]))

        bounding_box = resize_img[r_box[1]:r_box[3], r_box[0]:r_box[2]]

        resize_label = {}
        resize_label['path'] = ''
        resize_label['width'] = self.__face_size
        resize_label['height'] = self.__face_size
        resize_label['num'] = 1
        resize_label['dim'] = 2
        resize_label['annos'] = [(1, 0)]

        return (bounding_box, resize_label)


if __name__ == '__main__':
    root = '/home/zjq/dp_data_set/wider_face/Annotations'
    img_path = '/home/zjq/dp_data_set/wider_face/WIDER_train/images/'

    fi = FileIterator(AbsOp=TransferFace(face_size=32, img_path=img_path), total=1000)

    fi.iter_read(read_path=root)
