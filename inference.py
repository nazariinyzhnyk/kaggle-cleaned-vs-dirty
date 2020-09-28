import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms

from modeling.augmentation import get_dataloaders
from data_processing.utils import pjoin

transform_image = {
    'to_tensor_and_normalize': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# List of transformation methods
transforms_list = {
    'original': transforms.Compose([
        transforms.Resize((224, 224)),
    ]),
    'crop_180': transforms.Compose([
        transforms.CenterCrop(180),
        transforms.Resize((224, 224)),
    ]),
    'crop_160': transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize((224, 224)),
    ]),
    'crop_140': transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((224, 224)),
    ]),
    'gray_280': transforms.Compose([
        transforms.Grayscale(3),
        transforms.CenterCrop(280),
        transforms.Resize((224, 224)),
    ]),
    'gray_200': transforms.Compose([
        transforms.Grayscale(3),
        transforms.CenterCrop(200),
        transforms.Resize((224, 224)),
    ]),
    'r_crop_180_1': transforms.Compose([
        transforms.RandomCrop(180),
        transforms.Resize((224, 224)),
    ]),
    'r_crop_180_2': transforms.Compose([
        transforms.RandomCrop(180),
        transforms.Resize((224, 224)),
    ]),
    'r_crop_180_3': transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomCrop(180),
        transforms.Resize((224, 224)),
    ]),
}


def get_latest_model_name(pth):
    """If model name was not specified this function will find latest trained"""
    models = os.listdir(pth)
    models = [m for m in models if m.endswith('.pt')]
    if not models:
        raise RuntimeError('"models" folder is empty. train a model with calling main.py.')
    models.sort()
    return models[-1]


def inference(model_name='', model_save_dir='models', work_dir=pjoin('data', 'working_copy'), thresh=0.5):
    if not os.path.isdir(model_save_dir):
        raise RuntimeError('No "models" directory was detected. Run main.py to train a model.')

    if not model_name:  # if model name is blank - find latest model trained
        model_name = get_latest_model_name(model_save_dir)

    model_path = os.path.join(model_save_dir, model_name)
    print(f'Will load model within path: "{model_path}"')
    net = torch.load(model_path)
    _, _, test_dl = get_dataloaders(pjoin(work_dir, 'train_split'),
                                    pjoin(work_dir, 'valid_split'),
                                    pjoin(work_dir, 'test'))

    data = {'id': [],
            'label': []}
    for img_original, labels, img_id in tqdm(test_dl.dataset):
        probs = []
        img = os.path.split(img_id)[1].replace('.jpg', '')

        for i, method in enumerate(transforms_list):
            img_transformed = transforms_list[method](img_original)
            tensor = transform_image['to_tensor_and_normalize'](img_transformed)
            tensor = tensor.to('cpu')
            tensor = tensor.unsqueeze(0)

            with torch.set_grad_enabled(False):
                preds = net(tensor)

            probs.append(torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy()[0])
        label = 'dirty' if np.mean(probs) > 0.5 else 'cleaned'
        data['id'].append(img)
        data['label'].append(label)
        df = pd.DataFrame(data)
        df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    inference()
