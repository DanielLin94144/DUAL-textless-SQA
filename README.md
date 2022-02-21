# DUAL-textless-SQA
*This repo is under-constructed*
* Installation 
* Data Preparation 
    * Download our NMSQA dataset
        * [[link]](https://ntucc365-my.sharepoint.com/:u:/g/personal/r10942104_ntu_edu_tw/EZpxoRWns-NHoJnvaJERmDAB8WjHUf39obN4vQwQYHz73g?e=gU2GJi)
        * Dataset Usage
            * Directory format
                - train
                - dev
                - test
                    - test-SQuAD
                    - test-OOD 

            ![](https://i.imgur.com/vwuoTCH.png)

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
* Training 
    ```
    python train.py --exp_name [exp name] --config baseline.yaml
    ```
* Evaluation
    ```
    python evaluate.py --data_dir [data dir path] --model_path [model checkpoint dir] --output_dir [output dir path] --out_fname [output name]
    ```
* Results
* Citation
