from include import *


class AbsXmlDecoder(object):
    def __init__(self):
        pass

    def decode(self, label):
        pass


class XmlDecoderManager(object):
    def __init__(self, AbsXmlDecoder):
        self.__xml_decoder = AbsXmlDecoder

    def set_decoder(self, AbsXmlDecoder):
        self.__xml_decoder = AbsXmlDecoder

    def decode(self, label):
        return self.__xml_decoder.decode(label)


class WFXmlDecoder(AbsXmlDecoder):
    def decode(self, fp_label):
        """
         decode one wider face xml data to general object data
         :param label_file:
         :return:
         """
        assert fp_label
        print('aaa')
        """
        one anno format
        {'path':'...', 'width': 1024, 'high': 768, 'num': 10, 'dim': 4, 'anno': [[1, 2, 3, 4], [5, 6, 7, 8]]}

        """
        img_anno = {}
        count = 0

        tree = xml_et.parse(fp_label)
        root = tree.getroot()

        img_anno['path'] = label_path_to_img_path(root.find('filename').text)
        img_anno['width'] = int(root.find('size').find('width').text)
        img_anno['height'] = int(root.find('size').find('height').text)

        for _ in root.iter('object'):
            count += 1

        img_anno['num'] = count
        img_anno['dim'] = 4

        box_anno = []
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            box = (int(xmlbox.find('xmin').text),
                   int(xmlbox.find('ymin').text),
                   int(xmlbox.find('xmax').text),
                   int(xmlbox.find('ymax').text))
            box_anno.append(box)

        img_anno['annos'] = box_anno
        print(img_anno)
        return img_anno


if __name__ == '__main__':
    ins_decoder = XmlDecoderManager(WFXmlDecoder())

    # a.set_decoder(WFXmlDecoder())

    with open('/home/zjq/dp_data_set/wider_face/Annotations/0--Parade_0_Parade_marchingband_1_5.xml') as fp_label:
        ret = ins_decoder.decode(fp_label)
        print(ret)