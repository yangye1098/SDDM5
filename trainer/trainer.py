import torch
from utils import inf_loop, MetricTracker
import time
from datetime import timedelta
import torchaudio
from numpy import inf
from logger import TensorboardWriter


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, world_size, rank,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None):

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.device = device
        # distributed training
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # or debug purpose
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler



        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.valid_period = cfg_trainer.get('valid_period', 10)
        self.n_valid_data_batch = cfg_trainer.get('n_valid_data_batch', 2)
        self.log_step = cfg_trainer.get('log_step', 100)
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf


        # only loss for train
        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


        # audio sample dir
        sample_path = config.save_dir / 'samples'

        self.clean_path = sample_path / 'clean'
        self.output_path = sample_path / 'output'
        self.noisy_path = sample_path / 'noisy'

        if self.rank == 0:
            sample_path.mkdir(parents=True, exist_ok=True)
            self.clean_path.mkdir(parents=True, exist_ok=True)
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.noisy_path.mkdir(parents=True, exist_ok=True)


        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)



    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.is_distributed:
                self.data_loader.sampler.set_epoch(epoch)

            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            if self.rank == 0:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                except KeyError:
                    pass

                if not_improved_count > self.early_stop:
                    if self.rank == 0:
                        self.logger.info("Validation performance didn\'t improve for {} validation. "
                                     "Training stops.".format(self.early_stop))
                    break

            if self.rank == 0 and epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.epoch_start = time.time()
        self.model.train()
        self.train_metrics.reset()
        # if self.rank == 0:
        #     self.logger.debug('First Train batch.')
        #     batch_start = self.epoch_start

        for batch_idx, (clean_audio, noisy_audio, noisy_spec, _) in enumerate(self.data_loader):
            clean_audio, noisy_audio = clean_audio.to(self.device), noisy_audio.to(self.device)
            noisy_spec = noisy_spec.to(self.device)
            self.optimizer.zero_grad()
            # if self.rank == 0:
            #     self.logger.debug('Data loading: +{:f}s'.format(
            #     time.time() - batch_start))

            output, noise = self.model(clean_audio, noisy_audio, noisy_spec)
            # use noise in the loss function instead of clean_audio (y_0)
            loss = self.criterion(output, noise)

            # if self.rank == 0:
            #     self.logger.debug('Forward pass: +{:f}s'.format(
            #         time.time() - batch_start))

            loss.backward()
            # if self.rank == 0:
            #     self.logger.debug('Backword pass: +{:f}s'.format(
            #         time.time() - batch_start))

            self.optimizer.step()

            # if self.rank == 0:
            #     self.logger.debug('Optimization: +{:f}s'.format(
            #         time.time() - batch_start))

            if self.rank==0 and batch_idx>0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            # if self.rank == 0:
            #     self.logger.debug('Train batch {} started.'.format(
            #         batch_idx))
            #     batch_start = time.time()

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.rank == 0 and self.do_validation and (epoch % self.valid_period == 0):
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch, onld do validation if rank = 0

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.logger.debug('')
        self.logger.debug('Valid Epoch: {} started at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (clean_audio, noisy_audio, noisy_spec, _) in enumerate(self.valid_data_loader):
                if batch_idx >= self.n_valid_data_batch:
                    break

                clean_audio, noisy_audio = clean_audio.to(self.device), noisy_audio.to(self.device)
                noisy_spec = noisy_spec.to(self.device)

                # infer from noisy noisy_audioal input only
                if self.is_distributed:
                    output = self.model.module.infer(noisy_audio, noisy_spec)
                else:
                    output = self.model.infer(noisy_audio, noisy_spec)
                loss = self.criterion(output, clean_audio)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    temp = met(output, clean_audio)
                    self.valid_metrics.update(met.__name__, temp)

                # save the validation output
                for i in range(clean_audio.shape[0]):
                    torchaudio.save(self.output_path / f'{batch_idx}_{i}.wav', torch.unsqueeze(torch.squeeze(output[i,:,:]), 0).cpu(), self.config['sample_rate'])
                    torchaudio.save(self.clean_path / f'{batch_idx}_{i}.wav', torch.unsqueeze(torch.squeeze(clean_audio[i,:,:]), 0).cpu(), self.config['sample_rate'])
                    torchaudio.save(self.noisy_path / f'{batch_idx}_{i}.wav', torch.unsqueeze(torch.squeeze(noisy_audio[i,:,:]), 0).cpu(), self.config['sample_rate'])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')

        self.logger.debug('\nValid Epoch: {} finished at +{:.0f}s'.format(
            epoch, time.time()-self.epoch_start))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        lapsed = time.time() - self.epoch_start
        base = '[{}/{} | {:.0f}s/{}, ({:.0f}%), ]'
        current = batch_idx
        total = self.len_epoch

        time_left = lapsed * ((total/current) - 1)
        time_left = timedelta(seconds=time_left)
        return base.format(current, total, lapsed, time_left, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'distributed': self.is_distributed
        }
        if self.is_distributed:
            state['state_dict'] = self.model.module.state_dict()
        else:
            state['state_dict'] = self.model.state_dict()
        checkpoint_last = self.checkpoint_dir / 'checkpoint_current.pth'

        if checkpoint_last.is_file():
            checkpoint_last.rename(self.checkpoint_dir / 'checkpoint_last.pth')

        checkpoint_current = self.checkpoint_dir / 'checkpoint_current.pth'

        torch.save(state, checkpoint_current)
        self.logger.info("Saving checkpoint epoch {} as checkpoint_current.pth ...".format(epoch))
        if save_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)
            self.logger.info("Saving checkpoint epoch {} as model_best.pth ...".format(epoch))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded.")

        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
