import decord
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

decord.bridge.set_bridge('torch')
root = '/mnt/petrelfs/hukai/mmaction2/data/kinetics400'


class dataset(Data.Dataset):
    def __init__(self, mode: str = 'train') -> None:
        self.mode = mode
        label_file = '%s/kinetics400_%s_list_videos.txt' % (root, mode)
        prefix = '/dev/shm/kinetics400/videos_%s/' % mode

        with open(label_file) as f:
            f = [i.split() for i in f]
            f = [(prefix + i[0], int(i[1])) for i in f]
            self.datalist = f

        self.clip_len = 8
        self.frame_interval = 8
        self.covered_frames = (self.clip_len - 1) * self.frame_interval + 1

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224)])

    def __getitem__(self, index):
        path, label = self.datalist[index]
        video = decord.VideoReader(path)
        video = video.get_batch(self.get_clips(len(video)))
        video = video.permute(3, 0, 1, 2)
        video = self.transform(video)
        return video, label

    def __len__(self):
        return len(self.datalist)

    def get_clips(self, num_frames):
        max_offset = max(num_frames - self.covered_frames, 1)
        if self.mode == 'train':
            start = torch.randint(max_offset, size=[1]).item()
        else:
            start = max_offset // 2
        index = torch.arange(self.clip_len).mul(self.frame_interval)
        return (start + index).clamp_max(num_frames - 1)


def kinetics_loader(batch_size):
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
                                   num_workers=8,
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
