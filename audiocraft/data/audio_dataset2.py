# Modified from Audiocraft (https://github.com/facebookresearch/audiocraft)

"""AudioDataset support. In order to handle a larger number of files
without having to scan again the folders, we precompute some metadata
(filename, sample rate, duration), and use that to efficiently sample audio segments.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, fields
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
import logging
import os
from pathlib import Path
import random
import sys
import typing as tp

import torch
import torch.nn.functional as F
import numpy as np

from .audio import audio_read, audio_info
from .video import video_read_local, video_read_global
from .audio_utils import convert_audio
from .zip import PathInZip
import h5py
try:
    import dora
except ImportError:
    dora = None  # type: ignore


@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls) if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
            }


@dataclass(order=True)
class AudioMeta(BaseInfo):
    path: str
    video_path: str
    duration: float
    sample_rate: int
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None
    # info_path is used to load additional information about the audio file that is stored in zip files.
    info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        # print(f'dictionary:{dictionary}')
        # print(f'cls:{cls}')
        base = cls._dict2fields(dictionary)
        # print(f'base:{base}')
        if 'info_path' in base and base['info_path'] is not None:
            base['info_path'] = PathInZip(base['info_path'])
        # print(f'base:{base}')
        # exit()
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d['info_path'] is not None:
            d['info_path'] = str(d['info_path'])
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    seek_time: float
    # The following values are given once the audio is processed, e.g.
    # at the target sample rate and target number of channels.
    n_frames: int      # actual number of frames without padding
    total_frames: int  # total number of frames, padding included
    sample_rate: int   # actual sample rate
    channels: int      # number of audio channels.


DEFAULT_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

logger = logging.getLogger(__name__)


def _get_audio_meta(file_path: str, minimal: bool = True) -> AudioMeta:
    """AudioMeta from a path to an audio file.

    Args:
        file_path (str): Resolved path of valid audio file.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
    Returns:
        AudioMeta: Audio file path and its metadata.
    """
    info = audio_info(file_path)
    amplitude: tp.Optional[float] = None
    if not minimal:
        wav, sr = audio_read(file_path)
        amplitude = wav.abs().max().item()
    return AudioMeta(file_path, info.duration, info.sample_rate, amplitude)


def _resolve_audio_meta(m: AudioMeta, fast: bool = True) -> AudioMeta:
    """If Dora is available as a dependency, try to resolve potential relative paths
    in list of AudioMeta. This method is expected to be used when loading meta from file.

    Args:
        m (AudioMeta): Audio meta to resolve.
        fast (bool): If True, uses a really fast check for determining if a file
            is already absolute or not. Only valid on Linux/Mac.
    Returns:
        AudioMeta: Audio meta with resolved path.
    """
    def is_abs(m):
        if fast:
            return str(m)[0] == '/'
        else:
            os.path.isabs(str(m))

    if not dora:
        return m

    if not is_abs(m.path):
        m.path = dora.git_save.to_absolute_path(m.path)
    if m.info_path is not None and not is_abs(m.info_path.zip_path):
        m.info_path.zip_path = dora.git_save.to_absolute_path(m.path)
    return m


def find_audio_files(path: tp.Union[Path, str],
                     exts: tp.List[str] = DEFAULT_EXTS,
                     resolve: bool = True,
                     minimal: bool = True,
                     progress: bool = False,
                     workers: int = 0) -> tp.List[AudioMeta]:
    """Build a list of AudioMeta from a given path,
    collecting relevant audio files and fetching meta info.

    Args:
        path (str or Path): Path to folder containing audio files.
        exts (list of str): List of file extensions to consider for audio files.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
        progress (bool): Whether to log progress on audio files collection.
        workers (int): number of parallel workers, if 0, use only the current thread.
    Returns:
        list of AudioMeta: List of audio file path and its metadata.
    """
    audio_files = []
    futures: tp.List[Future] = []
    pool: tp.Optional[ThreadPoolExecutor] = None
    with ExitStack() as stack:
        if workers > 0:
            pool = ThreadPoolExecutor(workers)
            stack.enter_context(pool)

        if progress:
            print("Finding audio files...")
        for root, folders, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = Path(root) / file
                if full_path.suffix.lower() in exts:
                    audio_files.append(full_path)
                    if pool is not None:
                        futures.append(pool.submit(_get_audio_meta, str(audio_files[-1]), minimal))
                    if progress:
                        print(format(len(audio_files), " 8d"), end='\r', file=sys.stderr)

        if progress:
            print("Getting audio metadata...")
        meta: tp.List[AudioMeta] = []
        for idx, file_path in enumerate(audio_files):
            try:
                if pool is None:
                    m = _get_audio_meta(str(file_path), minimal)
                else:
                    m = futures[idx].result()
                if resolve:
                    m = _resolve_audio_meta(m)
            except Exception as err:
                print("Error with", str(file_path), err, file=sys.stderr)
                continue
            meta.append(m)
            if progress:
                print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


def load_audio_meta(path: tp.Union[str, Path],
                    resolve: bool = True, fast: bool = True) -> tp.List[AudioMeta]:
    """Load list of AudioMeta from an optionally compressed json file.

    Args:
        path (str or Path): Path to JSON file.
        resolve (bool): Whether to resolve the path from AudioMeta (default=True).
        fast (bool): activates some tricks to make things faster.
    Returns:
        list of AudioMeta: List of audio file path and its total duration.
    """
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'rb') as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        # print(f'line:{d}')
        m = AudioMeta.from_dict(d)
        # print(f'm:{m}')

        if resolve:
            m = _resolve_audio_meta(m, fast=fast)
            # print(f'm:{m}')
        meta.append(m)
        # exit() 
    # print(f'meta:{meta}')
    # exit()
    return meta


def save_audio_meta(path: tp.Union[str, Path], meta: tp.List[AudioMeta]):
    """Save the audio metadata to the file pointer as json.

    Args:
        path (str or Path): Path to JSON file.
        metadata (list of BaseAudioMeta): List of audio meta to save.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'wb') as fp:  # type: ignore
        for m in meta:
            json_str = json.dumps(m.to_dict()) + '\n'
            json_bytes = json_str.encode('utf-8')
            fp.write(json_bytes)


class AudioDataset:
    """Base audio dataset.

    The dataset takes a list of AudioMeta and create a dataset composed of segments of audio
    and potentially additional information, by creating random segments from the list of audio
    files referenced in the metadata and applying minimal data pre-processing such as resampling,
    mixing of channels, padding, etc.

    If no segment_duration value is provided, the AudioDataset will return the full wav for each
    audio file. Otherwise, it will randomly sample audio files and create a segment of the specified
    duration, applying padding if required.

    By default, only the torch Tensor corresponding to the waveform is returned. Setting return_info=True
    allows to return a tuple containing the torch Tensor and additional metadata on the segment and the
    original audio meta.

    Note that you can call `start_epoch(epoch)` in order to get
    a deterministic "randomization" for `shuffle=True`.
    For a given epoch and dataset index, this will always return the same extract.
    You can get back some diversity by setting the `shuffle_seed` param.

    Args:
        meta (list of AudioMeta): List of audio files metadata.
        segment_duration (float, optional): Optional segment duration of audio to load.
            If not specified, the dataset will load the full audio segment from the file.
        shuffle (bool): Set to `True` to have the data reshuffled at every epoch.
        sample_rate (int): Target sample rate of the loaded audio samples.
        channels (int): Target number of channels of the loaded audio samples.
        sample_on_duration (bool): Set to `True` to sample segments with probability
            dependent on audio file duration. This is only used if `segment_duration` is provided.
        sample_on_weight (bool): Set to `True` to sample segments using the `weight` entry of
            `AudioMeta`. If `sample_on_duration` is also True, the actual weight will be the product
            of the file duration and file weight. This is only used if `segment_duration` is provided.
        min_segment_ratio (float): Minimum segment ratio to use when the audio file
            is shorter than the desired segment.
        max_read_retry (int): Maximum number of retries to sample an audio segment from the dataset.
        return_info (bool): Whether to return the wav only or return wav along with segment info and metadata.
        min_audio_duration (float, optional): Minimum audio file duration, in seconds, if provided
            audio shorter than this will be filtered out.
        max_audio_duration (float, optional): Maximal audio file duration in seconds, if provided
            audio longer than this will be filtered out.
        shuffle_seed (int): can be used to further randomize
        load_wav (bool): if False, skip loading the wav but returns a tensor of 0
            with the expected segment_duration (which must be provided if load_wav is False).
        permutation_on_files (bool): only if `sample_on_weight` and `sample_on_duration`
            are False. Will ensure a permutation on files when going through the dataset.
            In that case the epoch number must be provided in order for the model
            to continue the permutation across epochs. In that case, it is assumed
            that `num_samples = total_batch_size * num_updates_per_epoch`, with
            `total_batch_size` the overall batch size accounting for all gpus.
    """
    def __init__(self,
                 meta: tp.List[AudioMeta],
                 segment_duration: tp.Optional[float] = None,
                 shuffle: bool = True,
                 num_samples: int = 10_000,
                 sample_rate: int = 48_000,
                 video_fps: int = 2,
                 video_overlap: int = 2,
                 if_add_gobal: bool = False,
                 global_mode: str = "average",
                 global_num_frames: int = 64,
                 global_feature_path: bool = False,
                 channels: int = 2,
                 pad: bool = True,
                 sample_on_duration: bool = True,
                 sample_on_weight: bool = True,
                 min_segment_ratio: float = 0.5,
                 max_read_retry: int = 10,
                 return_info: bool = False,
                 min_audio_duration: tp.Optional[float] = None,
                 max_audio_duration: tp.Optional[float] = None,
                 shuffle_seed: int = 0,
                 load_wav: bool = True,
                 permutation_on_files: bool = False,
                 ):
        assert len(meta) > 0, "No audio meta provided to AudioDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration
        self.meta: tp.List[AudioMeta] = self._filter_duration(meta)
        assert len(self.meta)  # Fail fast if all data has been filtered.
        self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            num_samples = len(self.meta)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.video_overlap = video_overlap
        self.if_add_gobal = if_add_gobal
        self.global_mode = global_mode
        self.global_num_frames = global_num_frames
        self.global_feature_path = global_feature_path
        self.channels = channels
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = return_info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: tp.Optional[int] = None
        self.load_wav = load_wav
        if not load_wav:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return self.num_samples

    def _get_sampling_probabilities(self, normalized: bool = True):
        """Return the sampling probabilities for each file inside `self.meta`."""
        scores: tp.List[float] = []
        for file_meta in self.meta:
            score = 1.
            if self.sample_on_weight and file_meta.weight is not None:
                score *= file_meta.weight
            if self.sample_on_duration:
                score *= file_meta.duration
            scores.append(score)
        probabilities = torch.tensor(scores)
        if normalized:
            probabilities /= probabilities.sum()
        return probabilities

    @staticmethod
    @lru_cache(16)
    def _get_file_permutation(num_files: int, permutation_index: int, base_seed: int):
        # Used to keep the most recent files permutation in memory implicitely.
        # will work unless someone is using a lot of Datasets in parallel.
        rng = torch.Generator()
        rng.manual_seed(base_seed + permutation_index)
        return torch.randperm(num_files, generator=rng)

    def sample_file(self, index: int, rng: torch.Generator) -> AudioMeta:
        """Sample a given file from `self.meta`. Can be overridden in subclasses.
        This is only called if `segment_duration` is not None.

        You must use the provided random number generator `rng` for reproducibility.
        You can further make use of the index accessed.
        """
        if self.permutation_on_files:
            assert self.current_epoch is not None
            total_index = self.current_epoch * len(self) + index
            permutation_index = total_index // len(self.meta)
            relative_index = total_index % len(self.meta)
            permutation = AudioDataset._get_file_permutation(
                len(self.meta), permutation_index, self.shuffle_seed)
            file_index = permutation[relative_index]
            return self.meta[file_index]

        if not self.sample_on_weight and not self.sample_on_duration:
            file_index = int(torch.randint(len(self.sampling_probabilities), (1,), generator=rng).item())
        else:
            file_index = int(torch.multinomial(self.sampling_probabilities, 1, generator=rng).item())

        return self.meta[file_index]

    def _audio_read(self, path: str, seek_time: float = 0, duration: float = -1):
        # Override this method in subclass if needed.
        
        if self.load_wav:
            return audio_read(path, seek_time, duration, pad=False)
        else:
            assert self.segment_duration is not None
            n_frames = int(self.sample_rate * self.segment_duration)
            return torch.zeros(self.channels, n_frames), self.sample_rate

    def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, SegmentInfo]]:
        if self.segment_duration is None:
            file_meta = self.meta[index]
            out, sr = audio_read(file_meta.path)
            out = convert_audio(out, sr, self.sample_rate, self.channels)
            n_frames = out.shape[-1]
            out = convert_audio(out, sr, self.sample_rate, self.channels)

            if self.if_add_gobal:
                # global_feature_path
                if self.global_feature_path!='' and os.path.exists(self.global_feature_path):
                    ytb_id = file_meta.video_path.split('/')[-1][:11]
                    with h5py.File(f'{self.global_feature_path}/{ytb_id}.h5', 'r') as file:
                        data = file['global_video_array'][:]
                        global_video = np.array(data)
                  #  local_video = video_read_local(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration)
                    local_video = video_read_local(file_meta.video_path,  target_fps=self.video_fps)

                else:
                    #local_video, global_video = video_read_global(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration, global_mode=self.global_mode, global_num_frames=self.global_num_frames)
                    local_video, global_video = video_read_global(file_meta.video_path,  target_fps=self.video_fps, global_mode=self.global_mode, global_num_frames=self.global_num_frames)
                        
            else: # local only
                video = video_read_local(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration)

            segment_info = SegmentInfo(file_meta, seek_time=0., n_frames=n_frames, total_frames=n_frames,
                                       sample_rate=self.sample_rate, channels=out.shape[0])
        else:
            rng = torch.Generator()
            if self.shuffle:
                # We use index, plus extra randomness, either totally random if we don't know the epoch.
                # otherwise we make use of the epoch number and optional shuffle_seed.
                if self.current_epoch is None:
                    rng.manual_seed(index + self.num_samples * random.randint(0, 2**24))
                else:
                    rng.manual_seed(index + self.num_samples * (self.current_epoch + self.shuffle_seed))
            else:
                # We only use index
                rng.manual_seed(index)

            for retry in range(self.max_read_retry):
                file_meta = self.sample_file(index, rng)
                # We add some variance in the file position even if audio file is smaller than segment
                # without ending up with empty segments

                overlap = self.video_overlap
                segment_duration_no_overlap = self.segment_duration - overlap
                max_seek = max(0, file_meta.duration - segment_duration_no_overlap * self.min_segment_ratio)
                max_value = max_seek 
                random_value = torch.rand(1, generator=rng).item() * max_value
                base_seek_time = segment_duration_no_overlap * int(random_value // segment_duration_no_overlap)                

                seek_time = random.randint(base_seek_time, base_seek_time + overlap)
                seek_time = min(max_seek, seek_time)
                
                try:
                    out, sr = audio_read(file_meta.path, seek_time, self.segment_duration, pad=False)

                    out = convert_audio(out, sr, self.sample_rate, self.channels)
                    n_frames = out.shape[-1]
                    target_frames = int(self.segment_duration * self.sample_rate)

                    if self.if_add_gobal:
                        if self.global_feature_path!='' and os.path.exists(self.global_feature_path):
                            ytb_id = file_meta.video_path.split('/')[-1][:11]
                            with h5py.File(f'{self.global_feature_path}/{ytb_id}.h5', 'r') as file:
                                data = file['global_video_array'][:]
                                global_video = np.array(data)
                            indices = np.linspace(0, global_video.shape[1]-1, num=self.global_num_frames, endpoint=True).round().astype(int)
                            global_video = global_video[:,indices,:,:]
                            global_video = torch.from_numpy(global_video)
                            local_video = video_read_local(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration)
                        else:
                            local_video, global_video = video_read_global(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration, global_mode=self.global_mode, global_num_frames=self.global_num_frames)
                        assert global_video.shape[1]==self.global_num_frames
                        
                        n_frames_video = local_video.shape[1]

                    else: # local only
                        video = video_read_local(file_meta.video_path,  target_fps=self.video_fps, seek_time=seek_time, duration=self.segment_duration)
                        n_frames_video = video.shape[1]

                    target_frames_video = int(self.segment_duration * self.video_fps)

                    if self.pad:
                        out = F.pad(out, (0, target_frames - n_frames))

                    segment_info = SegmentInfo(file_meta, seek_time, n_frames=n_frames, total_frames=target_frames,
                                               sample_rate=self.sample_rate, channels=out.shape[0])
                except Exception as exc:
                    logger.warning("Error opening file %s: %r", file_meta.path, exc)
                    if retry == self.max_read_retry - 1:
                        raise
                else:
                    break
        if self.if_add_gobal:
            if self.return_info:
                # Returns the wav and additional information on the wave segment
                return out, [local_video, global_video], segment_info
            else:
                return out, [local_video, global_video]
        else:
            if self.return_info:
                # Returns the wav and additional information on the wave segment
                return out, [video], segment_info
            else:
                return out, [video]

    def collater2(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert self.pad, "Must allow padding when batching examples of different durations."

        # In this case the audio reaching the collater is of variable length as segment_duration=None.
        to_pad = self.segment_duration is None and self.pad
        if to_pad:
            #max_len = max([wav.shape[-1] for wav, _ in samples])
            max_len = max([wav.shape[-1] for wav, *rest in samples])

            def _pad_wav(wav):
                return F.pad(wav, (0, max_len - wav.shape[-1]))

        if self.return_info:
            if len(samples) > 0:
                assert len(samples[0]) == 3
                assert isinstance(samples[0][0], torch.Tensor)
                assert isinstance(samples[0][1], list)
                assert isinstance(samples[0][2], SegmentInfo)


            wavs = [wav for wav, _, _ in samples]
            video_lists = [video_list for _, video_list, _ in samples]
            segment_infos = [copy.deepcopy(info) for _, _, info in samples]
            wav = torch.stack(wavs)
            
            assert isinstance(video_lists[0],list)
            if len(video_lists[0])==1:
                videos=[video_list[0] for video_list in video_lists]
                if to_pad:
                    # Each wav could be of a different duration as they are not segmented.
                    for i in range(len(samples)):
                        # Determines the total length of the signal with padding, so we update here as we pad.
                        segment_infos[i].total_frames = max_len
                        wavs[i] = _pad_wav(wavs[i])
                video = torch.stack(videos)
                
                return wav, [video], segment_infos
            
            elif len(video_lists[0])==2:

                local_videos=[video_list[0] for video_list in video_lists]
                global_videos=[video_list[1] for video_list in video_lists]

                if to_pad:
                    # Each wav could be of a different duration as they are not segmented.
                    for i in range(len(samples)):
                        # Determines the total length of the signal with padding, so we update here as we pad.
                        segment_infos[i].total_frames = max_len
                        wavs[i] = _pad_wav(wavs[i])
                local_video = torch.stack(local_videos)
                global_video = torch.stack(global_videos)

                return wav, [local_video, global_video], segment_infos    
                
        else:
            assert isinstance(samples[0], torch.Tensor)
            if to_pad:
                samples = [_pad_wav(s) for s in samples]
            return torch.stack(samples)

    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert self.pad, "Must allow padding when batching examples of different durations."

        # decide whether we need to pad to the max length
        to_pad = (self.segment_duration is None) and self.pad
        if to_pad:
            # compute max length over all wavs
            max_len = max(wav.shape[-1] for wav, *rest in samples)
            def _pad_wav(wav):
                return F.pad(wav, (0, max_len - wav.shape[-1]))

        if self.return_info:
            # sanity-check output of __getitem__
            assert len(samples[0]) == 3
            assert isinstance(samples[0][0], torch.Tensor)
            assert isinstance(samples[0][1], list)
            assert isinstance(samples[0][2], SegmentInfo)

            # unpack
            wavs         = [wav       for wav, _, _    in samples]
            video_lists  = [vlist     for _, vlist, _ in samples]
            segment_infos= [copy.deepcopy(info) for _, _, info in samples]

            # apply padding (and update total_frames)
            if to_pad:
                for i in range(len(wavs)):
                    wavs[i] = _pad_wav(wavs[i])
                    segment_infos[i].total_frames = max_len

            # now we can safely stack
            wav = torch.stack(wavs)

            # handle 1 or 2 videos per sample
            if len(video_lists[0]) == 1:
                videos = [vlist[0] for vlist in video_lists]
                video = torch.stack(videos)
                return wav, [video], segment_infos

            elif len(video_lists[0]) == 2:
                local_videos  = [vlist[0] for vlist in video_lists]
                global_videos = [vlist[1] for vlist in video_lists]
                local_video  = torch.stack(local_videos)
                global_video = torch.stack(global_videos)
                return wav, [local_video, global_video], segment_infos

        else:
            # simple case: just a list of wav tensors
            wavs = [s for s in samples]
            if to_pad:
                wavs = [_pad_wav(w) for w in wavs]
            return torch.stack(wavs)


    def _filter_duration(self, meta: tp.List[AudioMeta]) -> tp.List[AudioMeta]:
        """Filters out audio files with audio durations that will not allow to sample examples from them."""
        orig_len = len(meta)

        # Filter data that is too short.
        if self.min_audio_duration is not None:
            meta = [m for m in meta if m.duration >= self.min_audio_duration]

        # Filter data that is too long.
        if self.max_audio_duration is not None:
            meta = [m for m in meta if m.duration <= self.max_audio_duration]

        filtered_len = len(meta)
        removed_percentage = 100*(1-float(filtered_len)/orig_len)
        msg = 'Removed %.2f percent of the data because it was too short or too long.' % removed_percentage
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    @classmethod
    def from_meta(cls, root: tp.Union[str, Path], **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_dir():
            if (root / 'data.jsonl').exists():
                root = root / 'data.jsonl'
            elif (root / 'data.jsonl.gz').exists():
                root = root / 'data.jsonl.gz'
            else:
                raise ValueError("Don't know where to read metadata from in the dir. "
                                 "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
        meta = load_audio_meta(root)
        
        return cls(meta, **kwargs)

    @classmethod
    def from_path(cls, root: tp.Union[str, Path], minimal_meta: bool = True,
                  exts: tp.List[str] = DEFAULT_EXTS, **kwargs):
        """Instantiate AudioDataset from a path containing (possibly nested) audio files.

        Args:
            root (str or Path): Path to root folder containing audio files.
            minimal_meta (bool): Whether to only load minimal metadata or not.
            exts (list of str): Extensions for audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_file():
            meta = load_audio_meta(root, resolve=True)
        else:
            meta = find_audio_files(root, exts, minimal=minimal_meta, resolve=True)
        return cls(meta, **kwargs)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='audio_dataset',
        description='Generate .jsonl files by scanning a folder.')
    parser.add_argument('root', help='Root folder with all the audio files')
    parser.add_argument('output_meta_file',
                        help='Output file to store the metadata, ')
    parser.add_argument('--complete',
                        action='store_false', dest='minimal', default=True,
                        help='Retrieve all metadata, even the one that are expansive '
                             'to compute (e.g. normalization).')
    parser.add_argument('--resolve',
                        action='store_true', default=False,
                        help='Resolve the paths to be absolute and with no symlinks.')
    parser.add_argument('--workers',
                        default=10, type=int,
                        help='Number of workers.')
    args = parser.parse_args()
    meta = find_audio_files(args.root, DEFAULT_EXTS, progress=True,
                            resolve=args.resolve, minimal=args.minimal, workers=args.workers)
    save_audio_meta(args.output_meta_file, meta)


if __name__ == '__main__':
    main()
