import argparse
import torch
import torchaudio
from tqdm import tqdm

import data_loader as module_data
import model.loss as module_loss
import model.model as module_arch
import model.diffusion as module_diffusion
import model.network as module_network

from evaluate_results import evaluate
from data_loader import DataLoader
from parse_config import ConfigParser

torch.backends.cudnn.benchmark = True

def main(config):


    logger = config.get_logger('infer')

    # setup data_loader instances
    sample_rate = config['sample_rate']

    spec_transformer = config.init_obj('spec_transformer', module_data)
    infer_dataset = config.init_obj('infer_dataset', module_data, T=-1, sample_rate=sample_rate)
    infer_data_loader = DataLoader(infer_dataset, batch_size=1, num_workers=1, shuffle=False)

    logger.info('Finish initializing datasets')

    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = config.init_obj('diffusion', module_diffusion)
    network = config.init_obj('network', module_network)
    # prepare model for testing
    network = network.to(device)

    model = config.init_obj('arch', module_arch, diffusion, network)
    model.eval()
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)



    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])

    total_loss = 0.0

    sample_path = config.save_dir/'samples'
    sample_path.mkdir(parents=True, exist_ok=True)

    target_path = sample_path/'target'
    output_path = sample_path/'output'
    condition_path = sample_path/'condition'
    target_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    condition_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(infer_data_loader)
    with torch.no_grad():
        for i, (clean_audio, noisy_audio, idx) in tqdm(enumerate(infer_data_loader), desc='infer process', total=n_samples):
            clean_audio, noisy_audio = clean_audio.to(device).squeeze(0), noisy_audio.to(device).squeeze(0)
            # infer from conditional input only

            norm_factor = noisy_audio.abs().max().item()
            noisy_audio = noisy_audio / norm_factor
            # [1,1,N,L]
            noisy_spec = torch.unsqueeze(spec_transformer.spec_fwd(spec_transformer.stft(noisy_audio)), 0)

            T = noisy_spec.shape[-1]
            if T % 64 != 0:
                num_pad = 64 - T % 64
            else:
                num_pad = 0
            noisy_spec = torch.nn.functional.pad(noisy_spec, (0, num_pad, 0, 0), 'constant', 0)

            output = model.infer(noisy_spec)

            output = spec_transformer.istft(spec_transformer.spec_back(output.squeeze()),
                                                  clean_audio.shape[-1])
            output = output * norm_factor
            output = output.view((1, output.shape[-1]))  # [1, T]

            # save samples, or do something with output here

            name = infer_dataset.getName(idx)

            # stack back to full audio
            torchaudio.save(output_path/f'{name}.wav', output.cpu(), sample_rate)
            torchaudio.save(target_path/f'{name}.wav', clean_audio.cpu(), sample_rate)
            torchaudio.save(condition_path/f'{name}.wav', noisy_audio.cpu(), sample_rate)

            # computing loss, metrics on test set

            loss = loss_fn(output, clean_audio)
            total_loss += loss.item()

    log = {'loss': total_loss / n_samples}
    logger.info(log)

    # evaluate results
    metrics = {'pesq_wb', 'sisnr', 'stoi'}
    evaluate(sample_path, sample_rate, metrics, logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
