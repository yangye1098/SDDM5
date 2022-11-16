
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


def generate_inventory(path, file_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    return file_names


class AudioSpecDataset(Dataset):
    """
        load both audio waveform and spectrogram
    """
    def __init__(self, data_root, n_spec_frames, spec_config, sample_rate=8000, use_spec_pad=False):
        if spec_config['is_mel']:
            self.spec_type = '.mel.npy'
        else:
            self.spec_type = '.spec.npy'

        self.sample_rate = sample_rate
        # number of frame to load
        self.n_spec_frames = n_spec_frames
        self.spec_config = spec_config
        self.hop_samples = spec_config.get('hop_samples', 256)
        self.window_length = spec_config.get('window_length', 1024)
        self.use_spec_pad = use_spec_pad
        if self.use_spec_pad:
            self.spec_center = spec_config.get('center', True)
            self.spec_pad_mode = spec_config.get('pad_mode', "reflect")


        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))


        self.inventory = generate_inventory(self.clean_path, '.wav')
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        clean_audio, sr = torchaudio.load(self.clean_path/self.inventory[index])
        assert(sr == self.sample_rate)
        noisy_audio, sr = torchaudio.load(self.noisy_path/self.inventory[index])
        assert (sr == self.sample_rate)
        n_frames = clean_audio.shape[-1]
        assert (n_frames == noisy_audio.shape[-1])


        noisy_spec = torch.from_numpy(np.load(self.noisy_path/(self.getName(index)+self.spec_type)))

        if self.n_spec_frames >= 0:
            start = torch.randint(0, noisy_spec.shape[-1] - self.n_spec_frames, [1])
            end = start + self.n_spec_frames
        else:
            start = 0
            end = noisy_spec.shape[-1]

        if self.use_spec_pad:
            noisy_spec = noisy_spec[1:, start:end]
            if self.spec_center:
                half_frame = self.window_length//2
                start = start * self.hop_samples
                end = (end - 1) * self.hop_samples + self.window_length

                clean_audio = F.pad(clean_audio, (half_frame, half_frame), self.spec_pad_mode)
                noisy_audio = F.pad(noisy_audio, (half_frame, half_frame), self.spec_pad_mode)

            else:
                raise NotImplementedError

        else:
            noisy_spec = noisy_spec[:, start:end]
            start = start * self.hop_samples
            end = end * self.hop_samples

            if end > n_frames:
                clean_audio = F.pad(clean_audio, [0, end-n_frames], 'constant')
                noisy_audio = F.pad(noisy_audio, [0, end-n_frames], 'constant')

        clean_audio = clean_audio[:, start:end]
        noisy_audio = noisy_audio[:, start:end]

        return clean_audio, noisy_audio, noisy_spec, index

    def getName(self, idx):
        name = self.inventory[idx].rsplit('.', 1)[0]
        return name


class AudioDataset(Dataset):
    def __init__(self, data_root, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        # number of frame to load
        self.T = T

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))

        self.inventory = generate_inventory(self.clean_path, datatype)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index])
            assert(sr == self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index])
            assert (sr == self.sample_rate)
            n_frames = clean.shape[-1]
            assert (n_frames == noisy.shape[-1])

            if n_frames > self.T > 0:
                start_frame = torch.randint(0, n_frames - self.T, [1])
                clean = clean[:, start_frame:(start_frame+self.T)]
                noisy = noisy[:, start_frame:(start_frame+self.T)]

            elif self.T > n_frames > 0:
                clean = F.pad(clean, (0, self.T - n_frames), 'constant', 0)
                noisy = F.pad(noisy, (0, self.T - n_frames), 'constant', 0)

            elif self.T < 0:
                # no padding and no cut
                pass

        else:
            raise NotImplementedError



        return clean, noisy, index

    def getName(self, idx):
        name = self.inventory[idx].split('.', 1)[0]

        return name




class OutputDataset():
    def __init__(self, data_root, sample_rate=8000):
        self.sample_rate = sample_rate
        # number of frame to load

        self.clean_path = Path('{}/target'.format(data_root))
        self.noisy_path = Path('{}/condition'.format(data_root))
        self.output_path = Path('{}/output'.format(data_root))

        self.inventory = generate_inventory(self.output_path, '.wav')
        self.inventory = sorted(self.inventory)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        clean, sr = torchaudio.load(self.clean_path/self.inventory[index])
        assert(sr==self.sample_rate)
        noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index])
        assert (sr == self.sample_rate)
        output, sr = torchaudio.load(self.output_path/self.inventory[index])
        assert (sr == self.sample_rate)

        return clean, noisy, output

    def getName(self, idx):
        name = self.inventory[idx].rsplit('.', 1)[0]
        return name



if __name__ == '__main__':

    try:
        import simpleaudio as sa
        hasAudio = True
    except ModuleNotFoundError:
        hasAudio = False


