# from include_pkg import *
import os
from data_format_convert.label_data_protocol import *


class WiderFaceTransfer(object):
    def __init__(self, img_root_path='./',
                 read_label_path='./',
                 write_label_path='./'):
        self.__img_root_path = img_root_path
        self.__read_label_path = read_label_path
        self.__write_label_path = write_label_path

        self.__label_path_list = []

    def create_go_label(self):
        for (path, dirs, files) in os.walk(self.__img_root_path):
            """
            not involve the top dir
            """
            if files != []:
                for file in files:
                    img_path = os.path.join(self.__img_root_path, path.replace(self.__img_root_path, '').strip('/'), file)
                    label_path = os.path.join(self.__read_label_path, path.replace(self.__img_root_path, '').strip('/')
                                              + '_' + file.replace('.jpg', '.xml'))
                    self.__label_path_list.append([label_path, img_path])

        return WiderFaceProtocol().decode(self.__label_path_list)


if __name__ == '__main__':
    wft = WiderFaceTransfer(img_root_path='/home/zjq/dp_data_set/wider_face/WIDER_train/images',
                            read_label_path='/home/zjq/dp_data_set/wider_face/Annotations')

    go_label = wft.create_go_label()

    gop = GeneralObjectProtocol()

    gop.encode_txt_file(file_path='/home/zjq/dp_data_set/wider_face/go_label/train.txt', label_raw_data=go_label)
    # label_list = gop.decode_txt_file(file_path='/home/zjq/dp_data_set/wider_face/go_label/train.txt')

    # print(label_list[0:10])
