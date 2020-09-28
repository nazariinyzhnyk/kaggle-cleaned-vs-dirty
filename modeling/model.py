import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


class NNet(torch.nn.Module):
    def __init__(self, backbone=models.resnet152):
        super(NNet, self).__init__()

        self.net = backbone(pretrained=True)
        self.freeze_conv()
        self.modify_fc()

    def freeze_conv(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def modify_fc(self):
        fc_inputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_inputs, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
            # nn.LogSoftmax(dim=1) # For using NLLLoss()
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def save(self, pth):
        torch.save(self.net, pth)


def train_model(model, train_dataloader, valid_dataloader, nepoch):
    writer = SummaryWriter()
    print('\n' + model.__class__.__name__ + ' training with {} epochs started...\n'.format(nepoch))

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=1.0e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_hist = {'train': [], 'valid': []}
    accuracy_hist = {'train': [], 'valid': []}

    print('{:<7s}|{:^20s}|{:^20s}|'.format('', 'Train', 'Valid'))
    print('{:<7s}|{:>10s}{:>10s}|{:>10s}{:>10s}|'.format('Epoch', 'Loss', 'Acc', 'Loss', 'Acc'))
    print('-' * 50)
    for epoch in range(nepoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = train_dataloader
                if not epoch == 0:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, nepoch)
                writer.add_scalar('Accuracy/train', epoch_acc, nepoch)
                print('{:>3d}/{:>3d}|{:>10.4f}{:>10.4f}|'.format(epoch + 1, nepoch, epoch_loss, epoch_acc), end='')
            else:
                writer.add_scalar('Loss/validation', epoch_loss, nepoch)
                writer.add_scalar('Accuracy/validation', epoch_acc, nepoch)
                print('{:>10.4f}{:>10.4f}|'.format(epoch_loss, epoch_acc))

            loss_hist[phase].append(epoch_loss)
            accuracy_hist[phase].append(epoch_acc)

    return model, loss_hist, accuracy_hist
