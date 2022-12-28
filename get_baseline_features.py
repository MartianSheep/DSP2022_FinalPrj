import torch
import torchaudio
import numpy as np
import pickle
from tqdm import tqdm
from extractor import *
import yaml
import argparse

SAMPLE_RATE = 16000
MAX_DATA = 1000
MAX_LENGTH = 150000


def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)

    return wav.squeeze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    files = [{}, {}, {}]
    with open("./veri_test_class_slim.txt", "r") as f:
        for line in f.readlines():
            mode, file = line.split()
            mode = int(mode) - 1

            speaker, directory, wav_file = file.split("/")

            if int(speaker[3:]) < 270 or int(speaker[3:]) > 309:
                split = "dev"
            else:
                split = "test"

            if files[mode].get(speaker) is None:
                files[mode][speaker] = [f"{split}/wav/{speaker}/{directory}/{wav_file}"]
            elif len(files[mode][speaker]) < MAX_DATA:
                files[mode][speaker].append(
                    f"{split}/wav/{speaker}/{directory}/{wav_file}"
                )

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if "kaldi" in config:
        extracter, output_dim, frame_shift = get_extracter(config)
        downsample_rate = round(frame_shift * SAMPLE_RATE / 1000)
    else:
        exit()

    features = {}
    for speaker, fs in tqdm(files[0].items()):
        for f in fs:
            wav = reader(f"VoxCeleb1/{f}")
            if len(wav) > MAX_LENGTH:
                continue

            feature = torch.mean(extracter(wav), dim=0).numpy()

            if features.get(speaker) is None:
                features[speaker] = [feature]
            else:
                features[speaker].append(feature)

    with open(
        f"baseline_features/{config['kaldi']['feat_type']}80_feature_training.pickle",
        "wb",
    ) as f:
        pickle.dump(features, f)

    features = {}
    for speaker, fs in tqdm(files[2].items()):
        for f in fs:
            wav = reader(f"VoxCeleb1/{f}")
            if len(wav) > MAX_LENGTH:
                continue

            feature = torch.mean(extracter(wav), dim=0).numpy()

            if features.get(speaker) is None:
                features[speaker] = [feature]
            else:
                features[speaker].append(feature)

    with open(
        f"baseline_features/{config['kaldi']['feat_type']}80_feature_testing.pickle",
        "wb",
    ) as f:
        pickle.dump(features, f)
