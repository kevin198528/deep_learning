# from include_pkg import *
import os
from format_convert_pkg.label_data_protocol import *


class WiderFaceTransfer(object):
    def __init__(self, img_root_path='./',
                 read_label_path='./'):
        self.__img_root_path = img_root_path
        self.__read_label_path = read_label_path

        self.__label_path_list = []
        """
        label_data_list = [[img_file_path, (dim_x, dim_y), (hard_easy_flag, label_data), ...)], ...]
        """
        self.__label_data_list = []

    def decode_one_label(self, label_file):
        """
        decode one wider face xml data to general object data
        :param label_file:
        :return:
        """
        label_item = []
        count = 0
        label_item.append(label_file[1])
        with open(label_file[0]) as label_f:
            tree = xml_et.parse(label_f)
            root = tree.getroot()

            weight = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            label_item.append((weight, height))

            for _ in root.iter('object'):
                count += 1

            label_item.append((count, 5))

            for obj in root.iter('object'):
                difficult = int(obj.find('difficult').text)
                xmlbox = obj.find('bndbox')
                box = (difficult,
                       int(xmlbox.find('xmin').text),
                       int(xmlbox.find('ymin').text),
                       int(xmlbox.find('xmax').text),
                       int(xmlbox.find('ymax').text))
                label_item.append(box)

        return label_item

    def decode(self):
        """
        :param label_file_list:
        :param root_path:
        :return: label_data_list
        """
        for label_file in self.__label_path_list:
            self.__label_data_list.append(self.decode_one_label(label_file))

        return self.__label_data_list

    def encode(self):
        pass

    def create_gcp_data(self):
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

        return WiderFaceProtocol().decode()

if __name__ == '__main__':
    # wft = WiderFaceTransfer(img_root_path='/home/zjq/dp_data_set/wider_face/WIDER_train/images',
    #                         read_label_path='/home/zjq/dp_data_set/wider_face/Annotations')
    #
    # go_label = wft.create_go_label()

    ffc = FaceFrameCalibration()

    ffc_data = ffc.decode_txt_file(file_path='/home/zjq/dp_data_set/wider_face/go_label/train.txt')

    # ffc.encode_txt_file(file_path='/home/zjq/dp_data_set/wider_face/go_label/train1.txt', gcp_data=ffc_data)
