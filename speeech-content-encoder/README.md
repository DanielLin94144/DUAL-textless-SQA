# Speech Content Encoder (SCE)
Reference https://github.com/pytorch/fairseq/tree/master/examples/hubert/simple_kmeans 

## Dependency
* [s3prl](https://github.com/s3prl/s3prl)

## K-means model on LibriSpeech 100 hour 
We use the HuBERT-large model for feature extraction. The pre-trained K-means models are located in `km_100h_c128` and `km_100h_c500` folder for all 24 layers. 

## Speech2unit
`S2U_train_dev.py` and `S2U_test.py` encode NMSQA's raw waveforms into two features: 
1. `.code`: the discrete units
2. `.cnt`: the count of duplication for every discrete units

## If you want to train K-means model by yourself
### Calculate kmeans cluster
```shell
export FAIRSEQ_ROOT=~/fairseq/
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py ./train_100  train 1 0 ./mfcc_feat_100
# * This would shard the tsv file into ${nshard} and extract features for the ${rank}-th shard, where rank is an integer in [0, nshard-1]. Features would be saved at ${feat_dir}/${split}_${rank}_${nshard}.{npy,len}.

wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py ./train_100 train hubert_large_ll60k.pt 20 1 0 ./hubert_feat_100_layer_20;
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py ./hubert_feat_100_layer_20 train 1 ./km_feat_100_layer_20 500 --percent -1
```

### Validate clustering result
```shell
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py ./librisample/ --dest ./test_ds/ --valid-percent 0
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py ./test_ds/  train 1 0 ./mfcc_feat_test
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py ./test_ds train hubert_large_ll60k.pt 20 1 0 ./hubert_feat_test;
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py ./hubert_feat_test train ./km_feat_100_layer_20 1 0 ./lab_dir_test
```
* check whether `./lab_dir_test` result is the same as above.