from include import *


class Face24(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                       'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                       'target_img_path': '/home/zjq/dp_data_set/24x24_face/img',
                       'target_label_path': '/home/zjq/dp_data_set/24x24_face/label'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(CropFace(face_size=24, hue_flag=True))

        fi.set_coder(TxtCodeDecoder(div_num=100))

        # fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=100)


class Face0(object):
    def __init__(self):
        data_dict = {'src_img_path': '/home/zjq/dp_data_set/wider_face/WIDER_train/images/',
                     'src_label_path': '/home/zjq/dp_data_set/wider_face/Annotations/',
                     'target_img_path': '/home/zjq/dp_data_set/zero_face/img',
                     'target_label_path': '/home/zjq/dp_data_set/zero_face/label'}

        fi = FileIterator(path_dict=data_dict)

        fi.set_decoder(WFCodeDecoder())

        fi.set_transfer(ZeroFace(box_size=24, hue_flag=True))

        fi.set_coder(TxtCodeDecoder(div_num=100))

        # fi.set_coder(PickleCodeDecoder(div_num=5000))

        fi.iter_run(total=100)
