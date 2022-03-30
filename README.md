# DUAL-textless-SQA
*This repo is under-construction, please stay tuned for the update*

![](https://i.imgur.com/TCtkkp3.png)

This repository is the official implementation for [DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering](https://arxiv.org/abs/2203.04911) paper, and the release of the Natural Multi-speakers Spoken Question Answering **(NMSQA)** dataset. 

* Installation 
* Data Preparation 
    * Download our NMSQA dataset
        * [[dataset link]](https://ntucc365-my.sharepoint.com/:u:/g/personal/r10942104_ntu_edu_tw/EZpxoRWns-NHoJnvaJERmDAB8WjHUf39obN4vQwQYHz73g?e=gU2GJi)
        * [[Huggingface dataset link]](https://huggingface.co/datasets/voidful/NMSQA)
        * [[pretrained DUAL model on HuBERT-128]](https://ntucc365-my.sharepoint.com/:f:/g/personal/r10942104_ntu_edu_tw/EmDnNEHsnHlBiHNDnDzTGewB38uxBiimfrsY0EPgacP9oQ?e=OPp7hP)

        * Dataset Usage
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

                **NOTE**\\
                Current the spoken passage is split to segments of utterances. For the standard QA task, you should merge the segments back to the whole passages. The suffix of `-1`, `-2`, ..., `-n` is the segment number of specific passage.

    * Speech Content Encoder
    Please see details in `speeech-content-encoder`. 
    * Pre-process the QA labels 
    ```
    python code_answer.py
    ```
    
    [NOTE] Preprocessed data link (including passage merging and unit-level labels): [[link]](https://ntucc365-my.sharepoint.com/:f:/g/personal/r10942104_ntu_edu_tw/EqXPTZAQJcNGgWP0gLW0FngBmpWSPWEHZ0h-ukEbIleh3g?e=Qv4Bas)

* Training 
    ```
    python train.py --exp_name [exp name] --config baseline.yaml
    ```

* Evaluation
    ```
    python evaluate.py --data_dir [data dir path] --model_path [model checkpoint dir] --output_dir [output dir path] --out_fname [output name]
    ```

* Results
    | Discrete unit | PLM        | dev FF1 | dev AOS | test FF1 | test AOS |
    |---------------|------------|---------|---------|----------|----------|
    | HuBERT-64     | Longformer | 47.8    | 42.4    | 39.0     | 33.0     |
    | HuBERT-128    | Longformer | 54.2    | 48.5    | 56.0     | 49.1     |
    | HuBERT-512    | Longformer | 55.0    | 49.6    | 17.3     | 12.5     |
* Contact 
    Guan-Ting Lin (Email: daniel094144@gmail.com)

* Citation
    ```
    @article{lin2022dual,
    title={DUAL: Textless Spoken Question Answering with Speech Discrete Unit Adaptive Learning},
    author={Lin, Guan-Ting and Chuang, Yung-Sung and Chung, Ho-Lam and Yang, Shu-wen and Chen, Hsuan-Jui and Li, Shang-Wen and Mohamed, Abdelrahman and Lee, Hung-yi and Lee, Lin-shan},
    journal={arXiv preprint arXiv:2203.04911},
    year={2022}
    }
    ```

