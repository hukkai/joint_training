import torch

from model import I3DResNet
from tools import kinetics_loader

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def main():
    train_loader, _, val_loader = kinetics_loader(batch_size=64)

    mean = torch.tensor([123.7, 116.3, 103.5]).view(1, 3, 1, 1, 1)
    sttdev = torch.tensor([58.4, 57.1, 57.4]).view(1, 3, 1, 1, 1)

    mean = mean.cuda()
    sttdev = sttdev.cuda()

    model = I3DResNet(pretrain=None)
    params = torch.load('checkpoint/_166.t7', 'cpu')['backbone']
    model.load_state_dict(params)
    model = model.cuda()
    model.train()
    for video, _ in train_loader:
        video = video.cuda()
        video = video.float().sub_(mean).div_(sttdev)
        output = model(video)

    model.eval()
    correct = total = 0
    print('Begin Validation')
    for video, label in val_loader:
        video = video.cuda()
        video = video.float().sub_(mean).div_(sttdev)
        output = model(video)
        correct += output.argmax(1).cpu().eq(label).sum().item()
        total += label.size(0)
        print(100 * correct / total)

    print(correct, total)


if __name__ == '__main__':
    main()
