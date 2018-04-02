# from include import *
# # from img import *
# import random
#
#
# class AnnoCore(object):
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def decode_txt(file_path):
#         """
#
#         :param file_path:
#         :return:
#         """
#         assert file_path
#
#         annos = []
#         with open(file_path, 'r') as fp:
#             while True:
#                 ret = fp.readline().strip()
#                 if ret == '':
#                     return annos
#                 else:
#                     item_list = {}
#                     item_list['path'] = ret
#                     width, height = fp.readline().strip().split(' ')
#                     item_list['width'] = int(width)
#                     item_list['height'] = int(height)
#                     num, dim = fp.readline().strip().split(' ')
#                     item_list['num'] = int(num)
#                     item_list['dim'] = int(dim)
#
#                     anno_list = []
#                     for _ in range(int(num)):
#                         x_min, y_min, x_max, y_max = fp.readline().strip().split(' ')
#                         anno_list.append((int(x_min), int(y_min), int(x_max), int(y_max)))
#
#                     item_list['annos'] = anno_list
#
#                     annos.append(item_list)
#
#     @staticmethod
#     def encode_txt(file_path, annos):
#         """
#         :param file_path:
#         :param annos:
#         :return:
#         """
#         assert file_path
#         assert annos
#
#         with open(file_path, 'w') as fp:
#             for label in annos:
#                 fp.write(str(label['path']) + '\n')
#                 fp.write(str(label['width']) + ' ' + str(label['height']) + '\n')
#                 fp.write(str(label['num']) + ' ' + str(label['dim']) + '\n')
#                 for anno in label['annos']:
#                     fp.write(str(anno).replace(',', '').strip('(').strip(')') + '\n')
#
#
# class WiderFace(object):
#
#     def __init__(self):
#         """
#         label_data_list = [[img_file_path, (dim_x, dim_y), (hard_easy_flag, label_data), ...)], ...]
#         """
#         pass
#
#     @staticmethod
#     def decode1xml(img_file=None, label_file=None):
#         """
#         decode one wider face xml data to general object data
#         :param label_file:
#         :return:
#         """
#         assert img_file
#         assert label_file
#
#         """
#         one anno format
#         {'path':'...', 'width': 1024, 'high': 768, 'num': 10, 'dim': 4, 'anno': [[1, 2, 3, 4], [5, 6, 7, 8]]}
#
#         """
#         img_anno = {}
#         count = 0
#         img_anno['path'] = img_file
#         with open(label_file) as label_f:
#             tree = xml_et.parse(label_f)
#             root = tree.getroot()
#
#             img_anno['width'] = int(root.find('size').find('width').text)
#             img_anno['height'] = int(root.find('size').find('height').text)
#
#             for _ in root.iter('object'):
#                 count += 1
#
#             img_anno['num'] = count
#             img_anno['dim'] = 4
#
#             box_anno = []
#             for obj in root.iter('object'):
#                 xmlbox = obj.find('bndbox')
#                 box = (int(xmlbox.find('xmin').text),
#                        int(xmlbox.find('ymin').text),
#                        int(xmlbox.find('xmax').text),
#                        int(xmlbox.find('ymax').text))
#                 box_anno.append(box)
#
#             img_anno['annos'] = box_anno
#
#         return img_anno
#
#     @staticmethod
#     def decode_xml(img_root_path=None, anno_root_path=None):
#         """
#
#         :param img_root_path:
#         :param anno_root_path:
#         :return:
#         """
#         assert img_root_path
#         assert anno_root_path
#
#         annos = []
#         for (path, dirs, files) in os.walk(img_root_path):
#             """
#             not involve the top dir
#             """
#             if files != []:
#                 for file in files:
#                     """
#                     wider face protocol has bug with img path and anno path,
#                     the format is not the same
#                     """
#                     img_file = join(path, file)
#                     label_file = join(anno_root_path, path.replace(img_root_path, '').strip('/')
#                                               + '_' + file.replace('.jpg', '.xml'))
#
#                     anno = WiderFace.decode1xml(img_file, label_file)
#                     annos.append(anno)
#
#         return annos
#
#     @staticmethod
#     def encode_xml(self):
#         pass
#
#
# #
# # class Codecoder(object):
# #     def __init__(self):
# #         pass
#
#
#
#
#
# if __name__ == '__main__':
#     a = XmlDecoderManager(WFXmlDecoder())
#
#     # a.set_decoder(WFXmlDecoder())
#
#     a.decode_xml('root')
#     # pass
#     # img_root = '/home/zjq/dp_data_set/wider_face/WIDER_train/images'
#     # anno_root = '/home/zjq/dp_data_set/wider_face/Annotations'
#     #
#     # # annos = WiderFace.decode_xml(img_root_path=img_root,
#     # #                              anno_root_path=anno_root)
#     # # print(annos[0:10])
#     # # anno = WiderFace.decode1xml(img_file=join(img_root, '47--Matador_Bullfighter/47_Matador_Bullfighter_Matador_Bullfighter_47_734.jpg'),
#     # #                             label_file=join(anno_root, '47--Matador_Bullfighter_47_Matador_Bullfighter_Matador_Bullfighter_47_734.xml'))
#     # # AnnoCore.encode_txt(file_path='./anno.txt', annos=annos)
#     #
#     # annos = AnnoCore.decode_txt(file_path='./anno.txt')
#     #
#     # # print(annos[0:10])
#     # for item in annos[0:10]:
#     #     print(item)
#     #
#     # print(random.sample(annos[0:10], 1)[0])
