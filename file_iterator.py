import os
from code_decoder_manager import *
from transfer_manager import *
from s_utils import *


class FileIterator(object):
    def __init__(self, path_dict={}, total=100, div_num=10):
        """
        path_dict={'src_img_path','src_label_path','target_img_path','target_label_path'}
        """
        self.__total = total
        self.__path_dict = path_dict
        self.__div_num = int(total/div_num)


    def set_decoder(self, AbsCodeDecoder=None):
        self.__decoder = AbsCodeDecoder


    def set_coder(self, AbsCodeDecoder=None):
        self.__coder = AbsCodeDecoder


    def set_transfer(self, AbsTransfer=None):
        self.__transfer = AbsTransfer


    def iter_run(self):
        count = 0
        for root, dirs, files in os.walk(self.__path_dict['src_label_path']):
            for file in files:
                label_file = join(root, file)
                if self.__total <= 0:
                    return True

                try:
                    img_label = self.__decoder.decode(label_file, self.__path_dict)

                    img_label = self.__transfer.change(img_label)

                    group_id = int(count / self.__div_num)
                    img_id = int(count % self.__div_num)

                    img_label['label']['path'] = str(group_id) + '/' + str(img_id)

                    img_path = join(self.__path_dict['target_img_path'], str(group_id))

                    check_path(img_path)

                    label_path = join(self.__path_dict['target_label_path'], str(group_id))

                    check_path(label_path)

                    self.__coder.code(img_label, self.__path_dict)

                except TypeError as reason:
                    print('reason: ' + str(reason))
                    continue

                count += 1
                self.__total -= 1


if __name__ == '__main__':
    path_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                 'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                 'target_img_path': '/home/zjq/project/deep_learning/img',
                 'target_label_path': '/home/zjq/project/deep_learning/label'}

    fi = FileIterator(path_dict=path_dict, div_num=1)

    fi.set_decoder(WFCodeDecoder())

    fi.set_transfer(CropFace(face_size=300))

    fi.set_coder(TxtCodeDecoder())

    fi.iter_run()
