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
    def decode(self, file):
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
    def code(self, content, file):
        """
        :param file_path:
        :param annos:
        :return:
        """
        assert content
        assert file

        file.write(str(content['path']) + '\n')
        file.write(str(content['width']) + ' ' + str(content['height']) + '\n')
        file.write(str(content['num']) + ' ' + str(content['dim']) + '\n')
        for box in content['annos']:
            file.write(str(box).replace(',', '').strip('(').strip(')') + '\n')

    def decode(self, file):
        """
        :param file_path:
        :return:
        """
        assert file

        content = {}
        content['path'] = file.readline().strip()
        width, height = file.readline().strip().split(' ')
        content['width'] = int(width)
        content['height'] = int(height)
        num, dim = file.readline().strip().split(' ')
        content['num'] = int(num)
        content['dim'] = int(dim)

        anno_list = []
        for _ in range(int(num)):
            item = file.readline().strip().split(' ')
            anno_list.append(tuple(map(int, item)))

        content['annos'] = anno_list

        return content


class WFCodeDecoder(AbsCodeDecoder):
    def code(self, content, file):
        pass

    def decode(self, file):
        """
         decode one wider face xml data to general object data
         :param: label_file
         :return:
         """
        assert file

        """
        one anno format
        {'path':'...', 'width': 1024, 'high': 768, 'num': 10, 'dim': 4, 'anno': [[1, 2, 3, 4], [5, 6, 7, 8]]}

        """
        img_label = {}
        count = 0

        tree = xml_et.parse(file)
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

        return img_label


class JpgCodeDecoder(AbsCodeDecoder):
    def code(self, content, file):
        assert content
        assert file
        cv2.imwrite(file, content, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def decode(self, file):
        assert file
        return cv2.imread(file)


if __name__ == '__main__':
    ins_decoder = CodeDecoderManager()

    # ins_decoder.set_code_decoder(WFCodeDecoder())

    ins_decoder.set_code_decoder(TxtCodeDecoder())

    # file = '/home/zjq/dp_data_set/wider_face/Annotations/32--Worker_Laborer_32_Worker_Laborer_Worker_Laborer_32_585.xml'

    file = './test.txt'

    write_file = './write.txt'

    with open(file) as fp_label:
        ret = ins_decoder.decode(fp_label)
        # ins_decoder.set_code_decoder(TxtCodeDecoder())
        with open(write_file, 'w') as write_file:
            ins_decoder.code(ret, write_file)

        print(ret)