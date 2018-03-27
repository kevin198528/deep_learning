import os

join = lambda a, b: os.path.join(a, b)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)