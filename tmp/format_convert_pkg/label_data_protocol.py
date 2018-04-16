# from include_pkg import *
import xml.etree.ElementTree as xml_et


class GeneralCodecProtocol(object):
    def __init__(self):
        self.__label_list = []

    def decode_txt_file(self, file_path):
        with open(file_path, 'r') as fp:
            while True:
                ret = fp.readline().strip()
                if ret == '':
                    print('read file end')
                    return self.__label_list
                else:
                    item_list = []
                    """
                    add path
                    """
                    item_list.append(ret)
                    """
                    add weight hight
                    """
                    weight, hight = fp.readline().strip().split(' ')
                    item_list.append((int(weight), int(hight)))
                    """
                    add num dim
                    """
                    num, dim = fp.readline().strip().split(' ')
                    item_list.append((int(num), int(dim)))

                    for _ in range(int(num)):
                        dif, x_min, y_min, x_max, y_max = fp.readline().strip().split(' ')
                        item_list.append((int(dif), int(x_min), int(y_min), int(x_max), int(y_max)))

                    self.__label_list.append(item_list)

    def encode_txt_file(self, file_path, gcp_data):
        with open(file_path, 'w') as fp:
            for one_label in gcp_data:
                fp.write(str(one_label[0]) + '\n')
                fp.write(str(one_label[1]).replace(',', '').strip('(').strip(')') + '\n')
                fp.write(str(one_label[2]).replace(',', '').strip('(').strip(')') + '\n')
                for i in range(int(one_label[2][0])):
                    fp.write(str(one_label[3+i]).replace(',', '').strip('(').strip(')') + '\n')


class FaceFrameCalibration(GeneralCodecProtocol):
    pass
    # def convert_24x24_face(self, ffc_data):
    #     small_face_data =


class WiderFaceProtocol(object):

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

    def decode(self, label_path_list):
        """
        :param label_file_list:
        :param root_path:
        :return: label_data_list
        """
        for label_file in label_path_list:
            self.__label_data_list.append(self.decode_one_label(label_file))

        return self.__label_data_list

    def encode(self):
        pass
