from include import *


class Face24(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                       'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                       'target_img_path': '/home/zjq/dp_data_set/face24_25w_pickle/',
                       'target_label_path': '/home/zjq/dp_data_set/face24_25w_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(CropFace(face_size=24, hue_flag=True))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=250000)


class Face0(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                     'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                     'target_img_path': '/home/zjq/dp_data_set/face0_25w_pickle/',
                     'target_label_path': '/home/zjq/dp_data_set/face0_25w_pickle/'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(ZeroFace(box_size=24, hue_flag=True))

        # fi.set_coder(TxtCodeDecoder(div_num=100))

        fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=250000)


if __name__ == '__main__':
    face0 = Face0()
