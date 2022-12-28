import torch
import torchaudio

SAMPLE_RATE = 16000


def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)

    return wav.squeeze()


extractor = torch.hub.load("s3prl/s3prl", "hubert")
extractor.eval()
# if torch.cuda.is_available():
#     extractor = extractor.cuda()

wav1 = reader("./VoxCeleb1/dev/wav/id10001/zELwAz2W6hM/00008.wav")
# wave1 = wav1.cuda()
wav2 = reader("./VoxCeleb1/dev/wav/id10001/1zcIwhmdeo4/00002.wav")
# wave2 = wav2.cuda()

features = extractor([wav1, wav2])["hidden_state_3"]

print(features.shape)
# features2 = hubert_extractor([wave2])["hideen_state_3"]
