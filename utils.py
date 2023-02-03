import os


def mkdir(path):
    if not os.path.isdir(path):
        print('The directory is not present. Creating a new one..')
        os.mkdir(path)


def get_dir(path):
    return os.path.dirname(path)
