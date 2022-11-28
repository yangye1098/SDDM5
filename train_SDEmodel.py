import argparse
import collections
import torch
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.diffusion.sde as module_diffusion
import model.network as module_network
from parse_config import ConfigParser
from trainer import SDETrainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import os

torch.backends.cudnn.benchmark = True

def main(config):
    logger = config.get_logger('train')

    n_spec_frames = config['n_spec_frames']
    sample_rate =  config["sample_rate"]

    #
    # prepare distributed training
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    n_gpu = torch.cuda.device_count()
    # only work with 1 gpu per node
    assert n_gpu <= 1


    if world_size > 1:
        if 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
        else:
            raise ValueError
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
        # single process
        rank = 0
    if n_gpu >= 1:
        gpu_id = rank % n_gpu  # always 0 if n_gpu = 1
        device = torch.device(f'cuda:{gpu_id}' if n_gpu > 0 else 'cpu')
    else:
        device = 'cpu'

    if rank == 0:
        logger.info('Finish preparing gpu')

    # build model architecture, then print to console
    diffusion = config.init_obj('diffusion', module_diffusion)
    network = config.init_obj('network', module_network)
    network = network.to(device)

    model = config.init_obj('arch', module_arch, diffusion, network)

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id])
    else:
        # single process
        pass

    if rank == 0:
        logger.info(model)

    spec_transformer = config.init_obj('spec_transformer', module_data)
    # setup dataset
    tr_dataset = config.init_obj('tr_dataset', module_data, n_spec_frames=n_spec_frames, cspec_transformer=spec_transformer,
                                 sample_rate=config['sample_rate'])
    val_dataset = config.init_obj('val_dataset', module_data, T=-1,
                                  sample_rate=config['sample_rate'])
    if world_size > 1:
        # set up distributed data parallel
        tr_sampler = DistributedSampler(tr_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        tr_sampler = None

    tr_data_loader = config.init_obj('tr_data_loader', module_data, dataset=tr_dataset, sampler=tr_sampler)
    val_data_loader = config.init_obj('val_data_loader', module_data, dataset=val_dataset)

    if rank == 0:
        logger.info('Finish initializing datasets')

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lr_scheduler = None
    ema = None
    # config.init_obj('ema', torch_ema)

    trainer = SDETrainer(model, spec_transformer, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      world_size=world_size,
                      rank=rank,
                      data_loader=tr_data_loader,
                      valid_data_loader=val_data_loader,
                      ema=ema,
                      lr_scheduler=lr_scheduler
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Speech denoising diffusion model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
