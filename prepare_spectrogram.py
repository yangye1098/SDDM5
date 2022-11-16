import numpy as np
import torch
import torchaudio
from torchaudio import transforms as TT
from parse_config import ConfigParser

from glob import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def plot_grams(spec, sample_rate, frame_size):

    n_frame = spec.shape[1]
    freq_bins, time_bins = np.meshgrid(np.linspace(1, n_frame, num=n_frame),
                                       np.linspace(0, sample_rate/2, num=frame_size//2+1)
                                       )

    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')

    im = ax.pcolormesh(freq_bins, time_bins, spec)
    fig.colorbar(im, ax=ax)
    ax.set_title('Spectrogram')
    plt.show(block=True)


def main(config, dataset_type):
    sample_rate = config['sample_rate']

    spec_config = config['spectrogram']
    is_mel = spec_config['is_mel']
    window_length = spec_config['window_length']
    hop_samples = spec_config['hop_samples']

    center = spec_config['center']
    pad_mode = spec_config['pad_mode']

    path = config[dataset_type]['args']['data_root']
    filenames = glob(f'{path}/**/*.wav', recursive=True)

    power = 1.0
    normalize = True
    if is_mel:
        n_mel = spec_config['n_mel']
        spectrogram = TT.MelSpectrogram(n_fft=window_length,
                                        hop_length=hop_samples,
                                        window_fn=torch.hamming_window,
                                        f_min=20.0,
                                        f_max=sample_rate / 2.0,
                                        n_mels=n_mel,
                                        sample_rate=sample_rate,
                                        power=power,
                                        normalized=normalize,
                                        center=center,
                                        pad_mode=pad_mode
                                        )

    else:
        spectrogram = TT.Spectrogram(
                                     n_fft=window_length,
                                     hop_length=hop_samples,
                                     window_fn=torch.hamming_window,
                                     power=power,
                                     normalized=normalize,
                                     center=center,
                                     pad_mode=pad_mode
                                     )


    # multiprocess processing
    for i, filename in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        audio, sr = torchaudio.load(filename)
        assert (sr == sample_rate)

        spec = torch.squeeze(spectrogram(audio))
        # keep value in range [1e-4, 10]
        spec = torch.log10(spec) - 1
        if torch.max(spec) > 0:
            print(f'spec min: {torch.min(spec)}, max: {torch.max(spec)}')
        spec = torch.clamp((spec + 5) / 5, 0.0, 1.0)

        # plot_grams(spec, sample_rate, window_length)
        filename = filename.rsplit('.', 1)[0]
        if is_mel:
            np.save(f'{filename}.mel.npy', spec.cpu().numpy())
        else:
            np.save(f'{filename}.spec.npy', spec.cpu().numpy())


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Speech denoising diffusion model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('dataset_type', type=str,
                      help='dataset type: tr_dataset or val_dataset')

    args = args.parse_args()
    args.resume = None
    args.device = None

    config = ConfigParser.from_args(args)
    main(config, args.dataset_type )

