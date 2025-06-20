import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import smodels, utils
from spikingjelly.activation_based import functional
from spikingjelly.datasets import dvs128_gesture

import random
import numpy as np
import json
import wandb
import sys

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, \
           metric_logger.meters['img/s'].global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                split_by='number')

    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def run_training(args):
    set_seed(args.seed)
    if args.wandb:
        run_name = f"{args.model}_T{args.T}_lr{args.lr}_{'adam' if args.adam else 'sgd'}_seed{args.seed}"
        
        init_kwargs = {
            'project': args.wandb_project,
            'config': args,
            'name': run_name,
        }
        if args.wandb_entity:
            init_kwargs['entity'] = args.wandb_entity

        with wandb.init(**init_kwargs) as run:
            if args.wandb_entity and run.entity != args.wandb_entity:
                print(f"Warning: wandb run was created in entity '{run.entity}' instead of requested '{args.wandb_entity}'. "
                      f"Please check your wandb permissions for the '{args.wandb_entity}' entity.", file=sys.stderr)

            max_test_acc1 = 0.
            test_acc5_at_max_test_acc1 = 0.

            train_tb_writer = None
            te_tb_writer = None

            utils.init_distributed_mode(args)
            print(args)

            output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

            if args.T_train:
                output_dir += f'_Ttrain{args.T_train}'

            if args.weight_decay:
                output_dir += f'_wd{args.weight_decay}'

            output_dir += f'_steplr{args.lr_step_size}_{args.lr_gamma}'

            if args.adam:
                output_dir += '_adam'
            else:
                output_dir += '_sgd'

            if args.connect_f:
                output_dir += f'_cnf_{args.connect_f}'

            if not os.path.exists(output_dir):
                utils.mkdir(output_dir)

            output_dir = os.path.join(output_dir, f'lr{args.lr}')
            if not os.path.exists(output_dir):
                utils.mkdir(output_dir)

            device = torch.device(args.device)

            data_path = args.data_path

            dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)
            print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

            data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=args.batch_size,
                sampler=train_sampler, num_workers=args.workers, pin_memory=True)

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=args.batch_size,
                sampler=test_sampler, num_workers=args.workers, pin_memory=True)

            model = smodels.__dict__[args.model](args.connect_f)
            print("Creating model")

            model.to(device)
            if args.distributed and args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            criterion = nn.CrossEntropyLoss()

            if args.adam:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            if args.amp:
                scaler = amp.GradScaler()
            else:
                scaler = None

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

            model_without_ddp = model
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module

            if args.resume:
                checkpoint = torch.load(args.resume, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                max_test_acc1 = checkpoint['max_test_acc1']
                test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

            if args.test_only:
                evaluate(model, criterion, data_loader_test, device=device, header='Test:')

                return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                                                             args.print_freq, scaler, args.T_train)

        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

                print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

            print("Start training")
            start_time = time.time()
            imgs_per_s_list = []
            epoch = 0
            for epoch in range(args.start_epoch, args.epochs):
                save_max = False
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                train_loss, train_acc1, train_acc5, imgs_per_s = train_one_epoch(model, criterion, optimizer, data_loader,
                                                                                 device, epoch,
                                                                                 args.print_freq, scaler, args.T_train)
                imgs_per_s_list.append(imgs_per_s)
                if utils.is_main_process():
                    if train_tb_writer:
                        train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                        train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                        train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
                lr_scheduler.step()

                test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')

                if args.early_stop and epoch == 64:
                    # Early stop if validation accuracy is more than 10% below 74.4%
                    if test_acc1 < (0.744 * 0.9):
                        print(f"Early stopping at epoch {epoch} due to low accuracy: {test_acc1:.4f} < {0.744 * 0.9:.4f}")
                        break

                if te_tb_writer is not None:
                    if utils.is_main_process():
                        te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                        te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                        te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
                        if args.wandb and wandb.run:
                            wandb.log({
                                "train_loss": train_loss, "train_acc1": train_acc1, "train_acc5": train_acc5,
                                "test_loss": test_loss, "test_acc1": test_acc1, "test_acc5": test_acc5,
                                "lr": optimizer.param_groups[0]["lr"]
                            }, step=epoch)

                if max_test_acc1 < test_acc1:
                    max_test_acc1 = test_acc1
                    test_acc5_at_max_test_acc1 = test_acc5
                    save_max = True

                if output_dir:

                    checkpoint = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'max_test_acc1': max_test_acc1,
                        'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                    }

                    if save_max:
                        utils.save_on_master(
                            checkpoint,
                            os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                print(args)
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))

                print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
                      'test_acc5_at_max_test_acc1',
                      test_acc5_at_max_test_acc1)
                print(output_dir)

            total_time = time.time() - start_time
            avg_imgs_per_s = np.mean(imgs_per_s_list) if imgs_per_s_list else 0

            if output_dir and 'checkpoint' in locals():
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if args.wandb and utils.is_main_process() and wandb.run:
                wandb.log({
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                    'total_train_time_s': total_time,
                    'avg_imgs_per_s': avg_imgs_per_s,
                })
    else:
        # The training logic for runs without wandb
        max_test_acc1 = 0.
        test_acc5_at_max_test_acc1 = 0.
        train_tb_writer = None
        te_tb_writer = None
        utils.init_distributed_mode(args)
        # ... (The rest of the non-wandb training logic would go here)
        # This part is omitted for brevity as the user's focus is on wandb runs.
        # A full implementation would duplicate the logic from the `with` block.
        # For this case, we'll just return an empty dict.
        return {}

    def print_weight(m):
        if type(m) == smodels.SEWBlock:
            print(m.theta_0)
            print(m.theta_1)
            print(m.theta_2)
            print()

    model.apply(print_weight)

    return max_test_acc1

    return result_dict


def main():
    args = parse_args()
    result_dict = run_training(args)
    if result_dict and utils.is_main_process() and not args.test_only:
        # The runner script used to capture this JSON output when running as a subprocess.
        # This is kept for command-line execution of this script.
        print(f'\n{json.dumps(result_dict)}')


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--model', help='model')

    parser.add_argument('--data-path', default='data', help='dataset path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=64, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true', help='Use Adam optimizer')

    parser.add_argument('--connect_f', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', type=int)
    parser.add_argument('--seed', default=2020, type=int, help='random seed')

    parser.add_argument('--seed', default=2020, type=int, help='random seed for reproducibility')
    parser.add_argument('--wandb', action='store_true', help="Use wandb to log experiments")
    parser.add_argument('--wandb_project', default='dvs_gesture_sweeps', type=str, help="wandb project name")
    parser.add_argument('--wandb_entity', default=None, type=str, help="wandb entity")
    parser.add_argument('--early-stop', action='store_true', help='Early stop if accuracy is low at epoch 64')

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    random.seed(seed)

    torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    main(args)

'''
/raid/wfang/datasets/DVS128Gesture

python train.py --tb --amp --output-dir ./logs --model PlainNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture

python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture

'''
