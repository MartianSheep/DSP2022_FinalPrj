import torch
import torchaudio
import numpy as np
import pickle
from tqdm import tqdm

SAMPLE_RATE = 16000
MAX_DATA = 1000
MAX_LENGTH = 150000

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    
    return wav.squeeze()

if __name__ == '__main__':
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
                files[mode][speaker].append(f"{split}/wav/{speaker}/{directory}/{wav_file}")
    
    extractor = torch.hub.load("s3prl/s3prl", "hubert")
    extractor.eval()
    if torch.cuda.is_available():
        extractor = extractor.cuda()
    
    features = [{} for i in range(12)]
    for speaker, fs in tqdm(files[0].items()):
        for f in fs:
            wav = reader(f"VoxCeleb1/{f}")
            if len(wav) > MAX_LENGTH:
                continue
            wave = wav.cuda()
            
            hubert_features = extractor([wave])
            for i in range(12):
                feature = torch.mean(hubert_features[f"hidden_state_{i}"][0], dim=0).detach().cpu().numpy()
                
                if features[i].get(speaker) is None:
                    features[i][speaker] = [feature]
                else:
                    features[i][speaker].append(feature)
    
    for i in range(12):
        with open(f"hubert_feature/{i}_feature_training.pickle", "wb") as f:
            pickle.dump(features[i], f)
    
    features = [{} for i in range(12)]
    for speaker, fs in tqdm(files[2].items()):
        for f in fs:
            wav = reader(f"VoxCeleb1/{f}")
            if len(wav) > MAX_LENGTH:
                continue
            wave = wav.cuda()
            hubert_features = extractor([wave])
            for i in range(12):
                feature = torch.mean(hubert_features[f"hidden_state_{i}"][0], dim=0).detach().cpu().numpy()
                
                if features[i].get(speaker) is None:
                    features[i][speaker] = [feature]
                else:
                    features[i][speaker].append(feature)
    
    for i in range(12):
        with open(f"hubert_feature/{i}_feature_testing.pickle", "wb") as f:
            pickle.dump(features[i], f)
    