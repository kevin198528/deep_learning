# from include import *
from s_utils import *
import xml.etree.ElementTree as xml_et
import cv2
from s_utils import label_path_to_img_path


class AbsCodeDecoder(object):
    def __init__(self):
        pass

    def code(self, content, file):
        pass

    """
    return content
    """
    def decode(self, file, path):
        pass


class CodeDecoderManager(object):
    def __init__(self, AbsCodeDecoder=None):
        self.__code_decoder = AbsCodeDecoder

    def set_code_decoder(self, AbsCodeDecoder=None):
        self.__code_decoder = AbsCodeDecoder

    def code(self, content, file):
        return self.__code_decoder.code(content, file)

    def decode(self, file):
        return self.__code_decoder.decode(file)


class TxtCodeDecoder(AbsCodeDecoder):
    def __init__(self, div_num=10):
        self.__count = 0
        self.__div_num = div_num

    def code(self, img_label, path_dict):
        """
        :param file_path:
        :param annos:
        :return:
        """
        img = img_label['img']

        label = img_label['label']

        group_id = int(self.__count / self.__div_num)

        img_id = int(self.__count % self.__div_num)

        self.__count += 1

        label['path'] = str(group_id) + '/' + str(img_id)

        img_path = join(path_dict['target_img_path'], str(group_id))

        check_path(img_path)

        label_path = join(path_dict['target_label_path'], str(group_id))

        check_path(label_path)

        label_path = join(path_dict['target_label_path'], label['path']) + '.txt'
        img_path = join(path_dict['target_img_path'], label['path']) + '.jpg'

        with open(label_path, 'w') as fp:
            fp.write(str(label['path']) + '\n')
            fp.write(str(label['width']) + ' ' + str(label['height']) + '\n')
            fp.write(str(label['num']) + ' ' + str(label['dim']) + '\n')
            for box in label['annos']:
                fp.write(str(box).replace(',', '').strip('(').strip(')') + '\n')

        jpg_write(img, img_path)


    def decode(self, file):
        """
        :param file_path:
        :return:
        """
        content = {}
        anno_list = []

        with open(file, 'r') as fp:
            content['path'] = fp.readline().strip()
            width, height = fp.readline().strip().split(' ')
            content['width'] = int(width)
            content['height'] = int(height)
            num, dim = fp.readline().strip().split(' ')
            content['num'] = int(num)
            content['dim'] = int(dim)

            for _ in range(int(num)):
                item = fp.readline().strip().split(' ')
                anno_list.append(tuple(map(int, item)))

            content['annos'] = anno_list

        return content


class WFCodeDecoder(AbsCodeDecoder):
    def code(self, content, file):
        pass

    def decode(self, file, path):
        """
         decode one wider face xml data to general object data
         :param: label_file
         :return:
         """
        """
        one anno format
        {'path':'...', 'width': 1024, 'high': 768, 'num': 10, 'dim': 4, 'anno': [[1, 2, 3, 4], [5, 6, 7, 8]]}

        """
        img_label = {}
        count = 0

        with open(file) as fp_label:
            tree = xml_et.parse(fp_label)

        root = tree.getroot()

        img_label['path'] = label_path_to_img_path(root.find('filename').text)
        img_label['width'] = int(root.find('size').find('width').text)
        img_label['height'] = int(root.find('size').find('height').text)

        for _ in root.iter('object'):
            count += 1

        img_label['num'] = count
        img_label['dim'] = 4

        box_label = []
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            box = (int(xmlbox.find('xmin').text),
                   int(xmlbox.find('ymin').text),
                   int(xmlbox.find('xmax').text),
                   int(xmlbox.find('ymax').text))
            box_label.append(box)

        img_label['annos'] = box_label

        """
        img_path
        """
        img_file = join(path['src_img_path'], img_label['path'])

        img = jpg_read(img_file)
        if img is None:
            return False

        return {'img': img, 'label': img_label}


class PickleCodeDecoder(AbsCodeDecoder):
    def __init__(self, div_num=1000):
        self.__count = 0
        self.__div_num = div_num

    def code(self, img_label, path_dict):
        """
        :param file_path:
        :param annos:
        :return:
        """
        img = img_label['img']
        label = img_label['label']

        group_id = int(self.__count / self.__div_num)

        img_id = int(self.__count % self.__div_num)

        self.__count += 1

        label['path'] = str(group_id) + '/' + str(img_id)

        img_path = join(path_dict['target_img_path'], str(group_id))

        check_path(img_path)

        label_path = join(path_dict['target_label_path'], str(group_id))

        check_path(label_path)

        label_path = join(path_dict['target_label_path'], label['path']) + '.txt'
        img_path = join(path_dict['target_img_path'], label['path']) + '.jpg'

        with open(label_path, 'w') as fp:
            fp.write(str(label['path']) + '\n')
            fp.write(str(label['width']) + ' ' + str(label['height']) + '\n')
            fp.write(str(label['num']) + ' ' + str(label['dim']) + '\n')
            for box in label['annos']:
                fp.write(str(box).replace(',', '').strip('(').strip(')') + '\n')

        jpg_write(img, img_path)


    # data_w, lable_w = my_shuffle(data_w, lable_w)
    #
    # dic = {'data': data_w, 'lable': lable_w}
    #
    # j = pickle.dumps(dic)
    #
    # f = open('10w_1_zoom_and_bounding_box_pic_pickle_test', 'wb')  # 注意是w是写入str,wb是写入bytes,j是'bytes'
    # f.write(j)  # -------------------等价于pickle.dump(dic,f)
    #
    # f.close()



    def decode(self, file):
        """
        :param file_path:
        :return:
        """
        content = {}
        anno_list = []

        with open(file, 'r') as fp:
            content['path'] = fp.readline().strip()
            width, height = fp.readline().strip().split(' ')
            content['width'] = int(width)
            content['height'] = int(height)
            num, dim = fp.readline().strip().split(' ')
            content['num'] = int(num)
            content['dim'] = int(dim)

            for _ in range(int(num)):
                item = fp.readline().strip().split(' ')
                anno_list.append(tuple(map(int, item)))

            content['annos'] = anno_list

        return content

if __name__ == '__main__':

    arr = np.array([], np.int32)

    if arr.size is 0:
        print(0)


    a1 = np.array([[1, 1, 1]], np.int32)

    a2 = np.array([[2, 2, 2]], np.int32)

    a3 = np.array([[3, 3, 3]], np.int32)

    a = np.append(a, a1, axis=0)

    a = np.append(a, a2, axis=0)

    print(a)

    # pass
