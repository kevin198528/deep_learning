import xml.etree.ElementTree as xml_et
import numpy as np
import tensorflow as tf
import cv2
import os
import copy
import sys
import time
import pickle
import matplotlib.pyplot as plt
import re
import random
from data_process.code_decoder_manager import *
from data_process.file_iterator import *
from data_process.transfer_manager import *

from utils.s_utils import *

if __name__ == '__main__':
    # print(join('hello', 'world'))
    a = "47--Matador_Bullfighter_47_Matador_Bullfighter_matadorbullfighting_47_172.xml"

    b = re.findall(r'_\d{1,2}', a)[0]

    c = b.replace('_', '/')

    ret = a.replace(b, c, 1).replace('.xml', '.jpg')

    print(ret)


