import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from mmcv import imfrombytes
from mmengine import FileClient


class dataset(Data.Dataset):
    def __init__(self, mode: str = 'train'):
        label_file = '/mnt/cache/share/images/meta/%s.txt' % mode

        prefix = '/dev/shm/imagenet/%s/' % mode

        with open(label_file) as f:
            f = [i.split() for i in f]
            f = [(prefix + i[0], int(i[1])) for i in f]
            self.datalist = f

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224)])

        self.disk_client = FileClient(backend='disk')

    def __getitem__(self, index):
        path, label = self.datalist[index]
        image = self.disk_client.get(path)
        image = imfrombytes(image, flag='color', channel_order='rgb')
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.datalist)


def imagenet_loader(batch_size):
    train_dataset = dataset(mode='train')
    val_dataset = dataset(mode='val')

    train_sampler = val_sampler = None
    shuffle = True
    if torch.distributed.is_initialized():
        train_sampler = Data.distributed.DistributedSampler(train_dataset)
        val_sampler = Data.distributed.DistributedSampler(val_dataset)
        shuffle = False

    train_loader = Data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=12,
                                   shuffle=shuffle,
                                   drop_last=True,
                                   pin_memory=True,
                                   persistent_workers=True)

    val_loader = Data.DataLoader(val_dataset,
                                 batch_size=batch_size * 2,
                                 sampler=val_sampler,
                                 num_workers=8,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
    return train_loader, train_sampler, val_loader
