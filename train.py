import argparse
import os
import time
import uuid
from collections import deque
from typing import Optional

from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from models.backbone.base import Base as BackboneBase
from config.train_config import TrainConfig as Config
from data.base import Base as DatasetBase
from utils.logger import Logger as Log
from models.model import Model
from roi.wrapper import Wrapper as ROIWrapper


def _train(dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_checkpoints_dir: str,
           path_to_resuming_checkpoint: Optional[str]):
    # 根据使用的数据集的名字加载data文件夹中对应的.py文件
    dataset = DatasetBase.from_name(dataset_name)(path_to_data_dir, DatasetBase.Mode.TRAIN, Config.IMAGE_MIN_SIDE,
                                                  Config.IMAGE_MAX_SIDE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    Log.i('Found {:d} samples'.format(len(dataset)))

    # 根据backbone的名字加载对应的类
    backbone = BackboneBase.from_name(backbone_name)(pretrained=True)

    # rpn_pre_nms_top_n是在使用nms之前按照概率自高到低取rpn_pre_nms_top_n个anchor
    # rpn_post_nms_top_n是使用nms之后按照概率自高到低取rpn_post_nms_top_n个anchor
    model = Model(backbone, dataset.num_classes(), pooling_mode=Config.POOLING_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_scales=Config.ANCHOR_SCALES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    # 学习率自适应衰减，也就是当训练epoch达到milestones值时,初始学习率乘以gamma得到新的学习率;
    scheduler = MultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    should_stop = False

    # 每多少轮打印结果；每多少轮输出checkpoint
    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    # 如果传入了上一次训练的结果，将会从上一次训练的地方开始
    if path_to_resuming_checkpoint is not None:
        step = model.load(path_to_resuming_checkpoint, optimizer, scheduler)
        Log.i(f'Model has been restored from file: {path_to_resuming_checkpoint}')

    Log.i('Start training')

    while not should_stop:
        for batch_index, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            assert image_batch.shape[0] == 1, 'only batch size of 1 is supported'

            image = image_batch[0].cuda()
            bboxes = bboxes_batch[0].cuda()
            labels = labels_batch[0].cuda()

            forward_input = Model.ForwardInput.Train(image, gt_classes=labels, gt_bboxes=bboxes)
            forward_output: Model.ForwardOutput.Train = model.train().forward(forward_input)

            anchor_objectness_loss, anchor_transformer_loss, \
                proposal_class_loss, proposal_transformer_loss = forward_output
            loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            summary_writer.add_scalar('train/anchor_objectness_loss', anchor_objectness_loss.item(), step)
            summary_writer.add_scalar('train/anchor_transformer_loss', anchor_transformer_loss.item(), step)
            summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
            summary_writer.add_scalar('train/proposal_transformer_loss', proposal_transformer_loss.item(), step)
            summary_writer.add_scalar('train/loss', loss.item(), step)
            step += 1

            if step == num_steps_to_finish:
                should_stop = True

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                samples_per_sec = dataloader.batch_size * steps_per_sec
                eta = (num_steps_to_finish - step) / steps_per_sec / 3600
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                Log.i(f'[Step {step}] Avg. Loss = {avg_loss:.6f}, \
                Learning Rate = {lr:.6f} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)')

            if step % num_steps_to_snapshot == 0 or should_stop:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                Log.i(f'Model has been saved to {path_to_checkpoint}')

            if should_stop:
                break

    Log.i('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of '
                                                                                                          'dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of '
                                                                                                            'backbone '
                                                                                                            'model')
        parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
        parser.add_argument('-o', '--outputs_dir', type=str, default='./outputs', help='path to outputs directory')
        parser.add_argument('-r', '--resume_checkpoint', type=str, help='path to resuming checkpoint')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_scales', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SCALES))
        parser.add_argument('--pooling_mode', type=str, choices=ROIWrapper.OPTIONS, help='default: {.value:s}'.
                            format(Config.POOLING_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
        parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
        parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
        parser.add_argument('--step_lr_sizes', type=str, help='default: {!s}'.format(Config.STEP_LR_SIZES))
        parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
        parser.add_argument('--num_steps_to_display', type=int, help='default: {:d}'.
                            format(Config.NUM_STEPS_TO_DISPLAY))
        parser.add_argument('--num_steps_to_snapshot', type=int, help='default: {:d}'.
                            format(Config.NUM_STEPS_TO_SNAPSHOT))
        parser.add_argument('--num_steps_to_finish', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_FINISH))
        args = parser.parse_args()

        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_data_dir = args.data_dir
        path_to_outputs_dir = args.outputs_dir
        path_to_resuming_checkpoint = args.resume_checkpoint

        # 生成checkpoints的目录
        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, 'checkpoints-{:s}-{:s}-{:s}-{:s}'.format(
            time.strftime('%Y%m%d%H%M%S'), dataset_name, backbone_name, str(uuid.uuid4()).split('-')[0]))
        os.makedirs(path_to_checkpoints_dir)

        # 将接收到的超参数传入Config对象，并改变对象中初始化的默认值
        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_scales=args.anchor_scales, pooling_mode=args.pooling_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n,
                     learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                     step_lr_sizes=args.step_lr_sizes, step_lr_gamma=args.step_lr_gamma,
                     num_steps_to_display=args.num_steps_to_display, num_steps_to_snapshot=args.num_steps_to_snapshot,
                     num_steps_to_finish=args.num_steps_to_finish)

        Log.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
        Log.i('Arguments:')
        for k, v in vars(args).items():
            Log.i(f'\t{k} = {v}')
        Log.i(Config.describe())

        _train(dataset_name, backbone_name, path_to_data_dir, path_to_checkpoints_dir, path_to_resuming_checkpoint)

    main()
