import json, sys, re, unicodedata
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler, ConcatDataset
from tqdm import tqdm
from typing import List, Union

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
from f5_tts.logger import loggerConfig


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 1 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }

class CustomConcatDataset(ConcatDataset):
    def get_frame_len(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].get_frame_len(sample_idx)

    def _get_dataset_and_sample_index(self, idx):
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                if i == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.cumulative_sizes[i - 1]
                return i, sample_idx
        raise IndexError("Index out of range")

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False, logger: loggerConfig = None,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0
        self.logger = logger

        indices, batches = [], []
        data_source = self.sampler.data_source

        if self.logger:
            self.logger.add_info("DynamicBatchSampler", "Sorting with sampler... if slow, check whether dataset is provided with duration")
        for idx in self.sampler:
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        
        if self.logger:
            self.logger.add_info("DynamicBatchSampler", f"Creating dynamic batches with {frames_threshold} audio frames per gpu")
        for idx, frame_len in indices:
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset

def load_from_phonemize_txt(meta_path:str, wav_dir:str, logger: loggerConfig = None, vocab_map: dict = None) -> Dataset_:
    data = []
    meta_path = Path(meta_path)
    wav_dir = Path(wav_dir)
    WS_MAP = {              # ord(codepoint) → ' '
        ord('\u00A0'): ' ',   # NBSP
        ord('\u3000'): ' ',   # IDEOGRAPHIC SPACE
        ord('\t')    : ' '    # Horizontal TAB
    }
    DELETE_CHARS = '()[]{}<>'

    with open(meta_path, "r", encoding="utf-8", errors='ignore') as f:
        if logger:
            logger.add_info("load_from_phoenmize_txt", f'Parsing {meta_path.name}')
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            try:
                file_name, text, speaker, emotion, lang = line.split("|")
                text = text.translate({ord('\u00A0'): ' ', ord('\u3000'): ' ', ord('\t'): ' '})
                text = ''.join(' ' if unicodedata.category(c) == 'Zs' else c for c in text)
                text = re.sub(r' {2,}', ' ', text).strip()
                # 양쪽에 모두 " 가 있으면 제거
                if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                    text = text[1:-1].strip()
                    
                text = text.translate(str.maketrans('', '', DELETE_CHARS))
                text.strip()
            except ValueError:
                print(f"[WARN] Skipping malformed line: {line}")
                continue

            wav_path = wav_dir / file_name
            if not wav_path.exists():
                # print(f"[WARN] Missing wav file: {wav_path}")
                continue

            
            duration = torchaudio.info(str(wav_path)).num_frames / torchaudio.info(str(wav_path)).sample_rate

            try:
                info = torchaudio.info(str(wav_path))
                duration = info.num_frames / info.sample_rate
                if duration <= 1 or duration > 30:  # 너무 짧은 음성은 스킵
                    # print(f"[WARN] duration is to long: {wav_path}")
                    continue
                # if len(text) > 256:
                #     print(f"[WARN] text is to long: {wav_path}")
                #     continue
            except Exception as e:
                print(f"[ERROR] torchaudio.info failed on {wav_path}: {e}")
                continue
            
            if not text.strip():
                print(f"[WARN] Empty text for {file_name}, skipping")
                continue
            
            if vocab_map is not None:
                # (idx, ch) 튜플 리스트로 수집 → 미등록만 필터링
                unknown = [(i, ch) for i, ch in enumerate(text) if ch not in vocab_map]

                if unknown:
                    # 사람이 읽기 쉽게 "문자(위치)" 형태로 포맷
                    repr_unknown = ", ".join([f"'{ch}'(pos {i})" for i, ch in unknown])
                    # continue
                    raise ValueError(
                        f"[ERROR] 발견된 unknown token(s): [{repr_unknown}]\n"
                        f"파일: {file_name}\n"
                        f"문장: {text}\n"
                        "vocab.txt 를 갱신하거나 데이터에서 해당 문자를 제거하세요."
                    )
                    
            data.append({
                "audio_path": str(wav_path),
                "text": text,
                "speaker": speaker,
                "emotion": emotion,
                "lang": lang,
                "duration": duration
            })

    if len(data) == 0:
        raise ValueError(f"[ERROR] No valid data parsed from {meta_path}")
    
    return Dataset_.from_list(data)

def load_multiple_phonemize_datasets(
    root_paths: list[str],
    meta_paths: list[str],
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    logger: loggerConfig = None,
    vocab_map: dict = None,
) -> CustomDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")
    datasets = []

    for wav_root, metadata in zip(root_paths, meta_paths):
        meta_file = Path(metadata)
        wav_dir = Path(wav_root)
        print(f"[Loading] from: {meta_file}")

        # 1. load raw metadata
        hf_dataset = load_from_phonemize_txt(str(meta_file), str(wav_dir), logger, vocab_map)

        # durations = [None] * len(hf_dataset)
        # durations = [row["duration"] for row in hf_dataset]

        custom_dataset = CustomDataset(
            hf_dataset,
            durations=None,
            preprocessed_mel=False,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        datasets.append(custom_dataset)

    return CustomConcatDataset(datasets)


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
