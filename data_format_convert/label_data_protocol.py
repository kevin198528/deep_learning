# from include_pkg import *
import xml.etree.ElementTree as xml_et


class GeneralObjectProtocol(object):
    def __init__(self, save_path=None):
        self.__save_path = save_path

    # def tuple2str(self, tuple):
    #     for i in tuple:

    def decode(self):
        pass

    def encode_txt_file(self, label_raw_data):
        with open(self.__save_path, 'w') as w_f:
            w_f.write(label_raw_data[0][0] + ' ' + str(label_raw_data[0][1]) + '\n')

            for one_label in label_raw_data[1:]:
                w_f.write(str(one_label[0]) + '\n')
                w_f.write(str(one_label[1]) + '\n')
                w_f.write(str(one_label[2]) + '\n')
                for i in range(int(one_label[2][0])):
                    w_f.write(str(one_label[3+i]) + '\n')

class WiderFaceProtocol(object):
    """

    """
    def __init__(self):
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
            label_item.append((count, 4))

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

    def decode(self, label_path_list):
        """
        :param label_file_list:
        :param root_path:
        :return: label_data_list
        """
        data_len = len(label_path_list)
        name = 'wider_face'

        self.__label_data_list.append([name, data_len])

        for label_file in label_path_list:
            self.__label_data_list.append(self.decode_one_label(label_file))

        return self.__label_data_list

    def encode(self):
        pass
