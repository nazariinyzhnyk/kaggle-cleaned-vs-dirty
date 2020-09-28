import os
import shutil
import random

import numpy as np
import torch


def pjoin(*args) -> str:
    # Shortcut to os.path.join()
    return os.path.join(*args)


def create_working_copy(src: str, dest: str) -> None:
    """
    Creates working copy of src directory into destination directory

    Args:
        :param src: source directory to be copied
        :param dest: destination directory

    Returns:
        None
    """
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def train_val_split(data_root, train_dir='train_split', valid_dir='valid_split', class_names=('cleaned', 'dirty')):
    """Split train pictures to train and valid groups"""

    for dir_name in [train_dir, valid_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(data_root, dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        src_dir = os.path.join(data_root, 'train', class_name)
        files = os.listdir(src_dir)
        files = list(filter(lambda x: x.endswith('.jpg'), files))

        for i, file_name in enumerate(files):
            if i % 6 != 0:
                dst_dir = os.path.join(data_root, train_dir, class_name)
            else:
                dst_dir = os.path.join(data_root, valid_dir, class_name)
            shutil.copy(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))


def prepare_test_dir(src):
    """Moves files from src/test to src/test/unknown"""
    test_path = pjoin(src, 'test')
    os.makedirs(pjoin(test_path, 'unknown'), exist_ok=True)
    files = os.listdir(test_path)
    files = list(filter(lambda x: x.endswith('.jpg'), files))
    for f in files:
        shutil.move(pjoin(test_path, f), pjoin(test_path, 'unknown'))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clear_tb_logs(tb_dir='runs'):
    if os.path.isdir(tb_dir):
        shutil.rmtree(tb_dir)
