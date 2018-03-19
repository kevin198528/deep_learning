# from include_pkg import *
import xml.etree.ElementTree as xml_et


class GeneralObjectProtocol(object):
    def __init__(self):
        pass

    def decode_txt_file(self, file_path):
        label_list = []
        with open(file_path, 'r') as r_f:
            total = int(r_f.readline().strip())

            for _ in range(total):
                tmp_list = []
                path = r_f.readline().strip()
                weight, hight = r_f.readline().strip().split(' ')
                num, dim = r_f.readline().strip().split(' ')
                tmp_list.append(path)
                tmp_list.append((int(weight), int(hight)))
                tmp_list.append((int(num), int(dim)))
                for _ in range(int(num)):
                    dif, x_min, y_min, x_max, y_max = r_f.readline().strip().split(' ')
                    tmp_list.append((int(dif), int(x_min), int(y_min), int(x_max), int(y_max)))

                label_list.append(tmp_list)

        return label_list

    def encode_txt_file(self, file_path, label_raw_data):
        with open(file_path, 'w') as w_f:
            w_f.write(str(len(label_raw_data)) + '\n')

            for one_label in label_raw_data:
                w_f.write(str(one_label[0]) + '\n')
                w_f.write(str(one_label[1][0]) + ' ' + str(one_label[1][1]) + '\n')
                w_f.write(str(one_label[2][0]) + ' ' + str(one_label[2][1]) + '\n')
                for i in range(int(one_label[2][0])):
                    w_f.write(str(one_label[3+i][0]) + ' ' +
                              str(one_label[3+i][1]) + ' ' +
                              str(one_label[3+i][2]) + ' ' +
                              str(one_label[3+i][3]) + ' ' +
                              str(one_label[3+i][4]) + '\n')


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
