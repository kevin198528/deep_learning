import numpy as np
import pickle
import os
from utils.s_utils import *
import time
import random
import math


class BatchDataManage(object):
    def __init__(self, batch_num=50, file_path=[]):
        self.__batch_num = batch_num
        self.__file_path = file_path
        self.__file_io_list = self.get_file_io_list()

        # for class_data in self.__file_io_list:
        #     print(random.sample(class_data, 3))

        # print(self.__file_io_list[1])

        # time.sleep(100000)

        self.s_init()

    def s_init(self):
        self.__data, self.__label = self.fill_file_data()
        self.__data_len = self.__data.shape[0]
        self.__data_total = self.__data_len


    def get_file_io_list(self):
        class_list = []
        for class_path in self.__file_path:
            file_io_list = []
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    file_io_list.append(os.path.join(root, file))

            class_list.append(file_io_list)

        return class_list

    def fill_file_data(self):
        fill_data = np.array([], np.int32)
        fill_label = np.array([], np.float32)

        for class_data in self.__file_io_list:
            class_sample_data = random.sample(class_data, math.ceil(len(class_data)*0.2))

            for class_file in class_sample_data:
                with open(class_file, 'rb') as f_p:
                    record = pickle.loads(f_p.read())
                    data = record['data']
                    dim_1 = data.shape[1]
                    dim_2 = data.shape[2]
                    dim_3 = data.shape[3]
                    label = record['label']
                    data = np.reshape(data, [-1, dim_1 * dim_2 * dim_3])
                    mean = np.mean(data, axis=1).reshape(-1, 1)
                    std = np.std(data, axis=1).reshape(-1, 1)
                    data = (data - mean) / (std + 0.00001)
                    data = np.reshape(data, [-1, dim_1, dim_2, dim_3])

                    if fill_data.size == 0 and fill_label.size == 0:
                        fill_data = data
                        fill_label = label
                    else:
                        fill_data = np.append(fill_data, data, axis=0)
                        fill_label = np.append(fill_label, label, axis=0)

        fill_data = np.reshape(fill_data, [-1, dim_1*dim_2*dim_3])
        fill_data, fill_label = merge_shuffle(fill_data, fill_label)
        fill_data = np.reshape(fill_data, [-1, dim_1, dim_2, dim_3])

        return fill_data, fill_label

    def get_batch_data_label(self):
        self.__data_total -= self.__batch_num
        if self.__data_total < int(math.ceil(self.__data_len*0.9)):
            print('---')
            self.s_init()

        # idx_data = get_random_idx(self.__batch_num, 0, self.__data_len - self.__batch_num)
        idx_data = np.random.randint(0, self.__data_len - self.__batch_num, self.__batch_num)
        return self.__data[idx_data, :, :, :], self.__label[idx_data, :]


if __name__ == '__main__':
    # lst = [1, 2, 3, 4, 5]
    # print(random.sample(lst, 3))
    # print(lst)

    bdm = BatchDataManage(file_path=['/home/zjq/dp_data_set/face0_pickle', '/home/zjq/dp_data_set/face24_pickle'])

    for i in range(10000000):
        data, label = bdm.get_batch_data_label()
        print(data.shape)
        print(label.shape)
        print(i)
