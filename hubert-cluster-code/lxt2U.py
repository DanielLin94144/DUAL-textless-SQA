import numpy as np
import joblib
import torch
import torchaudio 
import pandas as pd
from tqdm import tqdm
import os 

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()


# train
df = pd.read_csv('/home/daniel094144/data/lxt_sqa/script.csv')
audio_file_dir = '/home/daniel094144/data/lxt_sqa/audio'

output_dir = '/home/daniel094144/data/SQA_code/w2v2_large_512/lxt_code'
extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2_large_ll60k')    
extractor.eval()

if torch.cuda.is_available():
        extractor = extractor.cuda()

apply_kmeans = ApplyKmeans('/home/daniel094144/hubert-cluster-code/km_feat_100_layer_14_512')

for file in tqdm(df['utterance_id'].values, desc='transforming lxt data to discrete code'):
    audio_file = os.path.join(audio_file_dir, file+'.wav')
    wavs = reader(audio_file)
    wavs = wavs.cuda()  

    if len(wavs) > 20 * SAMPLE_RATE: 
        print(f'{file} too long')
        chunks = torch.split(wavs, CHUNK_LENGTH) 
        for i, chunk in enumerate(chunks): 
            feat = extractor([chunk])
            feat = feat['hidden_state_14'].squeeze()
            
            if i == 0:
                feature = feat
            else: 
                feature = torch.cat([feature, feat], dim = 0)

        code = apply_kmeans(feature.cuda()) 
    else:
        feature = extractor([wavs])    

        
        code = apply_kmeans(feature['hidden_state_14'].squeeze().cuda())

    code = torch.tensor(code)

    merged_code, counts = torch.unique_consecutive(code, return_counts=True)
    np.savetxt(os.path.join(output_dir, file+'.code'), merged_code.long(), fmt='%i')    
    np.savetxt(os.path.join(output_dir, file+'.cnt'), counts.long(), fmt='%i')