import os
import time

from torchvision import models

from data_processing import pjoin, train_val_split, prepare_test_dir, set_seed, clear_tb_logs
from data_processing import create_working_copy, generate_rotated_images, remove_bg
from modeling import get_dataloaders, NNet, train_model

DATA_DIR = os.path.join('data', 'platesv2', 'plates')  # raw data from Kaggle
WORK_DIR = os.path.join('data', 'working_copy')  # directory we'll work with
TEST_PIPE = True

prep = {
    "rotation": True,
    "bg_removal": True,
    "clear_tb": True,
    "save_model": True
}

if TEST_PIPE:
    prep = {
        "rotation": False,
        "bg_removal": False,
        "clear_tb": True,
        "save_model": True
    }


def main():
    set_seed(42)
    if prep['clear_tb']:
        print('Clearing tensorboard logs directory')
        clear_tb_logs()

    print('Creating working copy of data directory')
    create_working_copy(DATA_DIR, WORK_DIR)

    if prep['rotation']:
        print('Generating rotated images')
        generate_rotated_images(pjoin(WORK_DIR, 'train', 'dirty'))
        generate_rotated_images(pjoin(WORK_DIR, 'train', 'cleaned'))

    if prep['bg_removal']:
        print('Removing background from images')
        remove_bg(pjoin(WORK_DIR, 'train', 'dirty'))
        remove_bg(pjoin(WORK_DIR, 'train', 'cleaned'))
        remove_bg(pjoin(WORK_DIR, 'test'))

    train_val_split(WORK_DIR)
    prepare_test_dir(WORK_DIR)

    train_dl, valid_dl, test_dl = get_dataloaders(pjoin(WORK_DIR, 'train_split'),
                                                  pjoin(WORK_DIR, 'valid_split'),
                                                  pjoin(WORK_DIR, 'test'))

    net = NNet(backbone=models.resnet18)

    model, losses, accuracies = train_model(net, train_dl, valid_dl, nepoch=10)
    model.eval()

    if prep['save_model']:
        if not os.path.isdir('models'):
            os.makedirs('models')
        pth = pjoin('models', str(round(time.time())) + '.pt')
        print(f'Saving trained model to {pth}')
        net.save(pth)


if __name__ == '__main__':
    main()
