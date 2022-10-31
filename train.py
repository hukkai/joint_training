import os
import time

import torch

from model import I3DResNet
from tools import imagenet_loader, init_DDP, kinetics_loader, lr_scheduler

torch.backends.cudnn.benchmark = False


def main():
    rank, local_rank, num_gpus = init_DDP(launcher='pytorch')
    device = torch.device('cuda')

    print('Inited distributed training!')

    num_images, num_videos = 1281167, 240435
    bs_video = 16
    num_gpus = 8
    num_iter = num_videos // (bs_video * num_gpus)
    bs_image = num_images // (num_iter * num_gpus)

    video_trainld, video_sampler, video_valld = kinetics_loader(bs_video)
    image_trainld, image_sampler, image_valld = imagenet_loader(bs_image)

    resume_path = None
    start_epoch, num_epochs, resume = 0, 200, False
    if resume_path is not None:
        ckpt = torch.load(resume_path, 'cpu')
        resume = True
        backbone_ckpt = ckpt['backbone']
        optimizer_ckpt = ckpt['optimizer']
        start_epoch = ckpt['start_epoch']
        current_iter = ckpt['current_iter']

    mean = torch.tensor([123.7, 116.3, 103.5]).view(1, 3, 1, 1)
    sttdev = torch.tensor([58.4, 57.1, 57.4]).view(1, 3, 1, 1)

    mean = mean.to(non_blocking=True, device=device)
    sttdev = sttdev.to(non_blocking=True, device=device)

    mean_v, sttdev_v = mean.unsqueeze(-1), sttdev.unsqueeze(-1)

    model = I3DResNet(pretrain='resnet50_3rdparty.pth')
    if resume:
        model.load_state_dict(backbone_ckpt)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      broadcast_buffers=False)

    param_norm = [p for p in model.parameters() if len(p.shape) < 2]
    param_conv = [p for p in model.parameters() if len(p.shape) >= 2]

    optimizer = torch.optim.SGD([{
        'params': param_norm,
        'weight_decay': 0.0
    }, {
        'params': param_conv,
        'weight_decay': 5e-5
    }],
                                lr=0.01 / 64 * num_gpus * bs_video,
                                momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    num_iters = min(len(video_trainld), len(image_trainld))
    scheduler = lr_scheduler(iter_per_epoch=num_iters,
                             max_epoch=num_epochs,
                             warmup_epoch=num_epochs // 10)

    if resume:
        optimizer.load_state_dict(optimizer_ckpt)
        scheduler.current_iter = current_iter
        scheduler.base_lr = optimizer_ckpt['param_groups'][0]['initial_lr']

    print('Begin Training')
    t = time.time()

    for epoch in range(start_epoch, num_epochs):

        model.train()
        video_sampler.set_epoch(epoch)
        image_sampler.set_epoch(epoch)

        video_iter = iter(video_trainld)
        image_iter = iter(image_trainld)

        correct_i = correct_v = total_i = total_v = 1e-9

        for idx in range(num_iter):
            optimizer.zero_grad()
            video_x, video_y = next(video_iter)
            image_x, image_y = next(image_iter)

            video_x = video_x.to(non_blocking=True, device=device)
            video_y = video_y.to(non_blocking=True, device=device)
            image_x = image_x.to(non_blocking=True, device=device)
            image_y = image_y.to(non_blocking=True, device=device)

            video_x = video_x.float().sub_(mean_v).div_(sttdev_v)
            image_x = image_x.float().sub_(mean).div_(sttdev)

            image_o = model(image_x)
            video_o = model(video_x)

            loss = criterion(image_o, image_y) + criterion(video_o, video_y)
            loss.backward()
            _ = scheduler.step(optimizer)
            optimizer.step()

            total_i += image_y.size(0)
            total_v += video_y.size(0)

            correct_i += image_o.argmax(1).eq(image_y).sum().item()
            correct_v += video_o.argmax(1).eq(video_y).sum().item()

        acc_i = 100. * correct_i / total_i
        acc_v = 100. * correct_v / total_v

        collect_info = [acc_i, acc_v]
        collect_info = torch.tensor(collect_info,
                                    dtype=torch.float32,
                                    device=device).clamp_min(1e-9)
        torch.distributed.all_reduce(collect_info)

        acc_i = collect_info[0] / num_gpus
        acc_v = collect_info[1] / num_gpus

        used = time.time() - t
        t = time.time()

        string = 'Epoch %d, image accuracy %.2f, video accuracy %.2f. ' \
                 'Time: %.2fmins.' % (
                    epoch, acc_i, acc_v, used/60)

        print(string)
        if rank == 0:
            state = {
                'backbone': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'start_epoch': epoch + 1,
                'current_iter': scheduler.current_iter
            }

            try:
                torch.save(state, 'checkpoint/_%d.t7' % epoch)
            except PermissionError:
                pass
            if epoch > 2:
                os.system('rm -f checkpoint/_%d.t7' % (epoch - 3))


if __name__ == '__main__':
    main()
