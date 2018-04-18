from include import *


class Face24test(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/wider_face/WIDER_train/images/',
                       'src_label_path': '/home/kevin/data_set/wider_face/Annotations/',
                       'target_img_path': '/home/kevin/data_set/face24_test_pickle/',
                       'target_label_path': '/home/kevin/data_set/face24_test_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(CropFace(face_size=24, hue_flag=False))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=5000)


class Face0test(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/wider_face/WIDER_train/images/',
                     'src_label_path': '/home/kevin/data_set/wider_face/Annotations/',
                     'target_img_path': '/home/kevin/data_set/face0_test_pickle/',
                     'target_label_path': '/home/kevin/data_set/face0_test_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(ZeroFace(box_size=24, hue_flag=False))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=5000)


class Face(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/src_img_data/wider_face/WIDER_train/images/',
                       'src_label_path': '/home/kevin/data_set/src_img_data/wider_face/Annotations/',
                       'target_img_path': '/home/kevin/data_set/pickle_img_data/test/face_pickle/',
                       'target_label_path': '/home/kevin/data_set/pickle_img_data/test/face_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(CropFace(face_size=36, hue_flag=True))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=1000))

        fi.iter_run(total=5000)


class Beijin(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/src_img_data/wider_face/WIDER_train/images/',
                     'src_label_path': '/home/kevin/data_set/src_img_data/wider_face/Annotations/',
                     'target_img_path': '/home/kevin/data_set/pickle_img_data/test/beijin_pickle/',
                     'target_label_path': '/home/kevin/data_set/pickle_img_data/test/beijin_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(ZeroFace(box_size=36, hue_flag=True))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=1000))

        fi.iter_run(total=5000)


class Wrj(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/src_img_data/wrj/img/',
                       'src_label_path': '/home/kevin/data_set/src_img_data/wrj/label',
                       'target_img_path': '/home/kevin/data_set/pickle_img_data/test/wrj_pickle',
                       'target_label_path': '/home/kevin/data_set/pickle_img_data/test/wrj_pickle'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WrjCodeDecoder())

        fi.set_transfer(CropWrj(face_size=36, hue_flag=True))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=900))

        fi.iter_run(total=900)


class Valid(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/kevin/data_set/src_img_data/valid/img/',
                       'src_label_path': '/home/kevin/data_set/src_img_data/valid/label',
                       'target_img_path': '/home/kevin/data_set/pickle_img_data/valid',
                       'target_label_path': '/home/kevin/data_set/pickle_img_data/valid'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(MyCodeDecoder())

        fi.set_transfer(CropFace(face_size=36, hue_flag=False))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=6))

        fi.iter_run(total=6)


if __name__ == '__main__':
    Valid()
    # a = Beijin()