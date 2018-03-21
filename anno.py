from include import *


class AnnoCore(object):
    def __init__(self):
        pass

    @staticmethod
    def decode_txt(file_path):
        """

        :param file_path:
        :return:
        """
        assert file_path

        annos = []
        with open(file_path, 'r') as fp:
            while True:
                ret = fp.readline().strip()
                if ret == '':
                    return annos
                else:
                    item_list = []
                    item_list.append(ret)
                    weight, hight = fp.readline().strip().split(' ')
                    item_list.append((int(weight), int(hight)))
                    num, dim = fp.readline().strip().split(' ')
                    item_list.append((int(num), int(dim)))

                    for _ in range(int(num)):
                        dif, x_min, y_min, x_max, y_max = fp.readline().strip().split(' ')
                        item_list.append((int(dif), int(x_min), int(y_min), int(x_max), int(y_max)))

                    annos.append(item_list)

    @staticmethod
    def encode_txt(file_path, annos):
        """

        :param file_path:
        :param annos:
        :return:
        """
        assert file_path
        assert annos

        with open(file_path, 'w') as fp:
            for label in annos:
                fp.write(str(label[0]) + '\n')
                fp.write(str(label[1]).replace(',', '').strip('(').strip(')') + '\n')
                fp.write(str(label[2]).replace(',', '').strip('(').strip(')') + '\n')
                for i in range(int(label[2][0])):
                    fp.write(str(label[3+i]).replace(',', '').strip('(').strip(')') + '\n')


class WiderFace(object):

    def __init__(self):
        """
        label_data_list = [[img_file_path, (dim_x, dim_y), (hard_easy_flag, label_data), ...)], ...]
        """
        pass

    @staticmethod
    def decode1xml(img_file=None, label_file=None):
        """
        decode one wider face xml data to general object data
        :param label_file:
        :return:
        """
        assert img_file
        assert label_file

        label_item = []
        count = 0
        label_item.append(img_file)
        with open(label_file) as label_f:
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

    @staticmethod
    def decode_xml(img_root_path=None, anno_root_path=None):
        """

        :param img_root_path:
        :param anno_root_path:
        :return:
        """
        assert img_root_path
        assert anno_root_path

        annos = []
        for (path, dirs, files) in os.walk(img_root_path):
            """
            not involve the top dir
            """
            if files != []:
                for file in files:
                    """
                    wider face protocol has bug with img path and anno path,
                    the format is not the same
                    """
                    img_file = join(img_root_path, path.replace(img_root_path, '').strip('/'), file)
                    label_file = join(anno_root_path, path.replace(img_root_path, '').strip('/')
                                              + '_' + file.replace('.jpg', '.xml'))

                    anno = WiderFace.decode1xml(img_file, label_file)
                    annos.append(anno)

        return annos

    @staticmethod
    def encode_xml(self):
        pass


if __name__ == '__main__':
    pass
    # img_root = '/home/zjq/dp_data_set/wider_face/WIDER_train/images'
    # anno_root = '/home/zjq/dp_data_set/wider_face/Annotations'

    # annos = WiderFace.decode_xml(img_root_path=img_root,
    #                              anno_root_path=anno_root)

    # anno = WiderFace.decode1xml(img_file=join(img_root, '12--Group/12_Group_Group_12_Group_Group_12_2.jpg'),
    #                             label_file=join(anno_root, '12--Group_12_Group_Group_12_Group_Group_12_2.xml'))

    # AnnoCore.encode_txt(file_path='./anno.txt', annos=[anno])

    # annos = AnnoCore.decode_txt(file_path='./anno.txt')

    # print(annos)
