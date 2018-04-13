import os
from code_decoder_manager import *
from transfer_manager import *
from s_utils import *


class FileIterator(object):
    def __init__(self, path_dict={}):
        """
        path_dict={'src_img_path','src_label_path','target_img_path','target_label_path'}

        """
        for value in path_dict.values():
            if os.path.exists(value) is False:
                print('path_dict is empty')
                assert False

        self.__path_dict = path_dict

    def set_decoder(self, AbsCodeDecoder=None):
        self.__decoder = AbsCodeDecoder


    def set_coder(self, AbsCodeDecoder=None):
        self.__coder = AbsCodeDecoder


    def set_transfer(self, AbsTransfer=None):
        self.__transfer = AbsTransfer


    def iter_run(self, total=100):
        total_in = total
        for root, dirs, files in os.walk(self.__path_dict['src_label_path']):
            for file in files:
                label_file = join(root, file)
                if total_in <= 0:
                    print('total_in < 0')
                    return True

                try:
                    img_label = self.__decoder.decode(label_file, self.__path_dict)

                    img_label = self.__transfer.change(img_label)

                    self.__coder.code(img_label, self.__path_dict)

                except TypeError as reason:
                    print('reason: ' + str(reason))
                    continue
                total_in -= 1


if __name__ == '__main__':
    # face24_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
    #                'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
    #                'target_img_path': '/home/zjq/dp_data_set/24x24_face/img',
    #                'target_label_path': '/home/zjq/dp_data_set/24x24_face/label'}

    face_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                  'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                  'target_img_path': '/home/zjq/dp_data_set/zero_face/img',
                  'target_label_path': '/home/zjq/dp_data_set/zero_face/label'}

    # face0_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
    #               'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
    #               'target_img_path': '/home/zjq/dp_data_set/zero_face/img',
    #               'target_label_path': '/home/zjq/dp_data_set/pickle_test'}

    # face24_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
    #               'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
    #               'target_img_path': '/home/zjq/dp_data_set/zero_face/img',
    #               'target_label_path': '/home/zjq/dp_data_set/face_24/'}

    fi = FileIterator(path_dict=face_dict)

    fi.set_decoder(WFCodeDecoder())

    # fi.set_transfer(CropFace(face_size=24, hue_flag=True))

    fi.set_transfer(ZeroFace(box_size=24, hue_flag=True))

    fi.set_coder(TxtCodeDecoder(div_num=100))

    # print(4)
    # fi.set_coder(PickleCodeDecoder(div_num=5000))

    fi.iter_run(total=100)

