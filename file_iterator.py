import os
from code_decoder_manager import *
from transfer_manager import *
from s_utils import *


class AbsOp(object):
    def __init__(self):
        pass

    def write_op(self):
        pass

    def read_op(self, file, num):
        pass


class TransferFace(AbsOp):
    def __init__(self, face_size, img_path):
        self.__face_size = face_size
        self.__img_path = img_path

    def read_op(self, file, num):
        wf_decoder = CodeDecoderManager(WFCodeDecoder())

        face24 = TransferManager(CropFace(face_size=self.__face_size))

        ret = wf_decoder.decode(file)
        # ins_decoder.set_code_decoder(TxtCodeDecoder())

        jpg_file = join(self.__img_path, ret['path'])

        img = jpg_read(jpg_file)
        if img is None:
            return False

        ret = face24.change(img, ret)
        if ret is False:
            return False

        # r_img, r_label
        print(ret[1])
        jpg_write(ret[0], './test/face' + str(num) + '.jpg')
        return True


class FileIterator(object):
    def __init__(self, path_dict={}, total=10, div_num=100):
        """
        path_dict={'src_img_path','src_label_path','target_img_path','target_label_path'}
        """
        self.__total = total
        self.__path_dict = path_dict

    def set_decoder(self, AbsCodeDecoder=None):
        self.__decoder = AbsCodeDecoder

    def set_coder(self, AbsCodeDecoder=None):
        self.__coder = AbsCodeDecoder

    def set_transfer(self, AbsTransfer=None):
        self.__transfer = AbsTransfer

    def iter_read(self):
        count = 0
        for root, dirs, files in os.walk(self.__path_dict['src_label_path']):
            for file in files:
                label_file = join(root, file)
                if self.__total <= 0:
                    return True

                # decode label (wider face for example)
                label = self.__decoder(label_file)

                img = jpg_read(jpg_file)
                if img is None:
                    return False

                ret = face24.change(img, ret)

                # transfer (crop 24 face for example)
                self.__transfer.change(img, label)

                if self.__op.read_op(file_path, self.__total):
                    self.__total -= 1
