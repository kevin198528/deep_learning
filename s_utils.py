# from include import *
import os
import numpy as np
import cv2
import re

join = lambda a, b: os.path.join(a, b)


def jpg_write(img, file):
    cv2.imwrite(file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def jpg_read(file):
    return cv2.imread(file)


def img_resize(img, attr, scale):
    return cv2.resize(img, (int(attr[0] * scale), int(attr[1] * scale)), interpolation=cv2.INTER_AREA)


def draw_boxes(img, boxes):
    """
    draw bounding_ box in raw_img

    """
    for box in boxes:
        print(box)
        n_box = np.array(box, np.int32)
        cv2.rectangle(img, tuple(n_box[0:2]), tuple(n_box[2:4]), (0, 255, 0), 1)


def show(img):
    """
    show img use opencv api

    """
    cv2.namedWindow('win', flags=0)
    cv2.imshow('win', img)
    cv2.waitKey(0)


def label_path_to_img_path(label_path):
    a = re.findall(r'_\d{1,2}', label_path)[0]

    b = a.replace('_', '/')

    return label_path.replace(a, b, 1).replace('.xml', '.jpg')


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def iou(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


if __name__ == '__main__':
    print(join('hello', 'world'))
    label_path = "47--Matador_Bullfighter_47_Matador_Bullfighter_matadorbullfighting_47_172.xml"

    print(label_path_to_img_path(label_path))

    # box = np.array([0, 0, 3, 3]).astype(np.float32)
    # boxes = np.array([[0, 0, 4, 4]]).astype(np.float32)
    # print(box)
    # print(boxes)
    # print(IoU(box, boxes))

    # box = np.array([0, 0, 2, 2])
    # boxes = np.array([[0, 0, 2, 2], [0, 0, 4, 4]])
    #
    # print(iou(box, boxes))

    # a = 24
    # b = 30
    # c = np.power(24, 2)
    # print(c/(a*a + b*b - c))

    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    #
    # c = [x + y for x, y in zip(a, b)]
    # print(c)
    # a = np.random.randint(0, 100, 10)
    # print(a)