export FAIRSEQ_ROOT=~/fairseq/
# python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py /home/daniel094144/data/LibriSpeech/LS100_fairseq  train 1 0 ./mfcc_feat_100_test
# * This would shard the tsv file into ${nshard} and extract features for the ${rank}-th shard, where rank is an integer in [0, nshard-1]. Features would be saved at ${feat_dir}/${split}_${rank}_${nshard}.{npy,len}.

# wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt

# wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt
# python3 $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py /home/daniel094144/data/LibriSpeech/LS100_fairseq train hubert_large_ll60k.pt 22 1 0 ./hubert_22;
# python3 $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_w2v2_feature.py /home/daniel094144/data/LibriSpeech/LS100_fairseq train libri960_big.pt 14 1 0 ./w2v2_feat_100_layer_14
# python3 $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py ./hubert_22 train 1 ./hubert_22_km_512_new 512 --percent -1
python3 $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py ./hubert_22 train 1 ./hubert_22_km_64_new 64 --percent -1
