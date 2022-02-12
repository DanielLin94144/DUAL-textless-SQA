# DUAL-textless-SQA

* Installation 
* Data Preparation 
    * Download our NMSQA dataset
    * Speech Quantization 
    Please see details in `hubert_cluster_code`
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
