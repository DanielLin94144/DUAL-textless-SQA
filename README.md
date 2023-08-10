# DUAL-textless-SQA
![](https://i.imgur.com/TCtkkp3.png)
This repository is the official implementation for [DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering](https://arxiv.org/abs/2203.04911) paper, and the release of the Natural Multi-speakers Spoken Question Answering **(NMSQA)** dataset. 

## Installation 
### Model
* [[Pretrained DUAL model on HuBERT-128]](https://drive.google.com/file/d/1BjbjM007q8zt1Z8cjGKCZps3E7beeSj2/view?usp=sharing)
### Dataset
Download the NMSQA dataset
* [[Original dataset]](https://ntucc365-my.sharepoint.com/:u:/g/personal/r10942104_ntu_edu_tw/EZpxoRWns-NHoJnvaJERmDAB8WjHUf39obN4vQwQYHz73g?e=gU2GJi)
* [[Parquet Format dataset]](https://github.com/DanielLin94144/DUAL-textless-SQA/tree/main/data.parquet)
* [[Huggingface Format dataset]](https://huggingface.co/datasets/voidful/NMSQA)

### Data Preparation for Original Dataset 
Preprocessed data link (including passage merging and unit-level labels, updated with question code): [[link]](https://ntucc365-my.sharepoint.com/:f:/g/personal/r10942104_ntu_edu_tw/EqXPTZAQJcNGgWP0gLW0FngBmpWSPWEHZ0h-ukEbIleh3g?e=Qv4Bas)

* Directory format
    - train
    - dev
    - test

* Files
    * For train and dev split
    `{split}-answer-span.csv`: answer time span in seconds
    `meta-{split}.csv: the duration`, speaker, and transcription of each utterance
    `{split}-textgrid.tar.gz`: force alignment of each utterance
    `{split}_audio.tar.gz`: utterance waveform files
    `{split}_hash2question.json`: map the hash value to question id
    * For test split
    `lxt_sqa.tar.gz`: contains all audio files in `audio` and transcriptions
    `meta-lxt.csv`: the duration, speaker, and transcription of each utterance
    `test/test-SQuAD/test-SQuAD-answer-span.csv`: the answer span in the test-SQuAD split
    `test/test-OOD/test-OOD-answer-span.csv`: the answer span in the test-OOD split

    **NOTE**
    Current the spoken passage is split to segments of utterances. For the standard QA task, you should merge the segments back to the whole passages. The suffix of `-1`, `-2`, ..., `-n` is the segment number of specific passage.

    * Speech Content Encoder
    Please see details in `speeech-content-encoder`. 
    * Pre-process the QA labels 
    ```
    python code_answer.py
    ```

### Parquet Format & Huggingface Format dataset
It basically follow the same file format as the Origin SQuAD with the following extra field:
```json=
{
   "id": Same as SQuAD,
   "title": Same as SQuAD,
   "context": Same as SQuAD,
   "question": Same as SQuAD,
   "answers":{
      "answer_start": Same as SQuAD,
      "audio_full_answer_end":[], Audio answer end position in second
      "audio_full_answer_start":[], Audio answer start position in second
      "audio_full_neg_answer_end":[], Audio answer end position in second that using the same words but not the correct one
      "audio_full_neg_answer_start":[], Audio answer start position in second that using the same words but not the correct one
      "audio_segment_answer_end":[],
      "audio_segment_answer_start":[],
      "text": Same as SQuAD
   },
   "content_segment_audio_path": Segment Audio Path,
   "content_full_audio_path": Complete Audio Path,
   "content_audio_sampling_rate": Audio Sampling Rate,
   "content_audio_speaker": Audio Speaker,
   "content_segment_text":"",
   "content_segment_normalized_text": Normalized Text for generating audio,
   "question_audio_path": Question Audio Path,
   "question_audio_sampling_rate": Audio Sampling Rate,
   "question_audio_speaker": Audio Speaker,
   "question_normalized_text": Normalized Text for generating audio,
}
```

## Training 
```
python train.py --exp_name [exp name] --config baseline.yaml
```

## Evaluation
```
python evaluate.py --data_dir [data dir path] --model_path [model checkpoint dir] --output_dir [output dir path] --out_fname [output name]
```

## Results
| Discrete unit | PLM        | dev FF1 | dev AOS | test FF1 | test AOS |
|---------------|------------|---------|---------|----------|----------|
| HuBERT-64     | Longformer | 47.8    | 42.4    | 39.0     | 33.0     |
| HuBERT-128    | Longformer | 54.2    | 48.5    | 56.0     | 49.1     |
| HuBERT-512    | Longformer | 55.0    | 49.6    | 17.3     | 12.5     |

# Contact 
Guan-Ting Lin (Email: daniel094144@gmail.com)
Eric Lam (Email: voidful.stack@gmail.com)

# Citation
```
@inproceedings{lin22c_interspeech,
  author={Guan-Ting Lin and Yung-Sung Chuang and Ho-Lam Chung and Shu-wen Yang and Hsuan-Jui Chen and Shuyan Annie Dong and Shang-Wen Li and Abdelrahman Mohamed and Hung-yi Lee and Lin-shan Lee},
  title={{DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={5165--5169},
  doi={10.21437/Interspeech.2022-612}
}
```


