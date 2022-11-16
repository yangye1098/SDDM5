
from os.path import join
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import torchaudio
import numpy as np
import torch.nn.functional as F
from pathlib import Path


def generate_inventory(path, file_type='.wav'):
    path = Path(path)

    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    return file_names


class CSpecTransformer():

    def __init__(
            self,
            n_fft=510, hop_length=128,
            spec_factor=0.15, spec_abs_exponent=0.5,
            transform_type="exponent"
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.windows = {}
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.transform_type = transform_type

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=length, center=True)




class CSpecDataset(Dataset):
    def __init__(self, data_root, n_spec_frames, cspec_transformer:CSpecTransformer, sample_rate=16000, normalize='noisy'):

        self.sample_rate = sample_rate
        # number of frame to load
        self.n_spec_frames = n_spec_frames
        self.cspec_transformer = cspec_transformer
        self.normalize = normalize

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))

        self.inventory = generate_inventory(self.clean_path, '.wav')
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        clean_audio, sr = torchaudio.load(self.clean_path / self.inventory[index])
        assert (sr == self.sample_rate)
        noisy_audio, sr = torchaudio.load(self.noisy_path / self.inventory[index])
        assert (sr == self.sample_rate)
        current_len = clean_audio.shape[-1]
        assert (current_len == noisy_audio.shape[-1])

        # cut audio
        # formula applies for center=True
        target_len = (self.n_spec_frames - 1) * self.cspec_transformer.hop_length
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            start = int(np.random.uniform(0, current_len-target_len))
            clean_audio = clean_audio[..., start:start+target_len]
            noisy_audio = noisy_audio[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            clean_audio = F.pad(clean_audio, (pad//2, pad//2+(pad%2)), mode='constant')
            noisy_audio = F.pad(noisy_audio, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = noisy_audio.abs().max()
        elif self.normalize == "clean":
            normfac = clean_audio.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        clean_audio = clean_audio / normfac
        noisy_audio = noisy_audio / normfac

        # transform to complex spectrogram
        clean_spec = self.cspec_transformer.stft(clean_audio)
        noisy_spec = self.cspec_transformer.stft(noisy_audio)

        clean_spec = self.cspec_transformer.spec_fwd(clean_spec)
        noisy_spec = self.cspec_transformer.spec_fwd(noisy_spec)
        return clean_spec, noisy_spec
