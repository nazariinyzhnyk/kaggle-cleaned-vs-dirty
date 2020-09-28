import torch
import torchvision
from torchvision import transforms

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),
        transforms.RandomChoice([transforms.CenterCrop(180),
                                 transforms.CenterCrop(160),
                                 transforms.CenterCrop(140),
                                 transforms.CenterCrop(120),
                                 transforms.Compose([transforms.CenterCrop(280),
                                                     transforms.Grayscale(3)]),
                                 transforms.Compose([transforms.CenterCrop(200),
                                                     transforms.Grayscale(3),
                                                     ]),
                                 ]),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(hue=(0.1, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),
        transforms.RandomChoice([transforms.CenterCrop(180),
                                 transforms.CenterCrop(160),
                                 transforms.CenterCrop(140),
                                 transforms.CenterCrop(120),
                                 transforms.Compose([transforms.CenterCrop(280),
                                                     transforms.Grayscale(3)]),
                                 transforms.Compose([transforms.CenterCrop(200),
                                                     transforms.Grayscale(3)]),
                                 ]),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(hue=(0.1, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])}


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dataloaders(train_dir, valid_dir, test_dir, batch_size=4):
    dataset = {
        'train': torchvision.datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
        'valid': torchvision.datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
        'test': ImageFolderWithPaths(test_dir, transform=None),
    }

    train_dataloader = torch.utils.data.DataLoader(dataset['train'],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=batch_size)

    valid_dataloader = torch.utils.data.DataLoader(dataset['valid'],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=batch_size)

    test_dataloader = torch.utils.data.DataLoader(dataset['test'],
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0)

    return train_dataloader, valid_dataloader, test_dataloader
