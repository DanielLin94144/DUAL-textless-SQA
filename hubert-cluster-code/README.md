# hubert-cluster-code
Reference https://github.com/pytorch/fairseq/tree/master/examples/hubert/simple_kmeans   

## Usage: Extract hubert code from clustering result
`wget https://raw.githubusercontent.com/voidful/hubert-cluster-code/main/km_feat_100/km_feat_100_layer_20`

```python
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(ds["speech"][0], return_tensors="pt").input_values  
hidden_states = model(input_values,output_hidden_states=True).hidden_states


import numpy as np
import joblib
import torch

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
            
apply_kmeans = ApplyKmeans('./km_100h_c500/km_feat_layer_22')
apply_kmeans(hidden_states[22].squeeze().cuda())
```

or using asrp
```python
import asrp

hc = asrp.HubertCode("facebook/hubert-large-ll60k", './km_100h_c500/km_feat_layer_22', 22)
hc('voice file path')
```

## Calculate kmeans cluster
```shell
export FAIRSEQ_ROOT=~/fairseq/
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py ./train_100  train 1 0 ./mfcc_feat_100
# * This would shard the tsv file into ${nshard} and extract features for the ${rank}-th shard, where rank is an integer in [0, nshard-1]. Features would be saved at ${feat_dir}/${split}_${rank}_${nshard}.{npy,len}.

wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py ./train_100 train hubert_large_ll60k.pt 20 1 0 ./hubert_feat_100_layer_20;
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py ./hubert_feat_100_layer_20 train 1 ./km_feat_100_layer_20 500 --percent -1
```

## Validate clustering result
```shell
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py ./librisample/ --dest ./test_ds/ --valid-percent 0
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py ./test_ds/  train 1 0 ./mfcc_feat_test
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py ./test_ds train hubert_large_ll60k.pt 20 1 0 ./hubert_feat_test;
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py ./hubert_feat_test train ./km_feat_100_layer_20 1 0 ./lab_dir_test
```
* check whether `./lab_dir_test` result is the same as above.