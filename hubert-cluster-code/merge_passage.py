import json
import torch
import numpy as np 
from tqdm import tqdm

with open('/home/daniel094144/E2E-SpokenQA/dev_segment_id.json', 'r') as f:
    segment_dict = json.load(f)

data_dir = '/home/daniel094144/data/SQA_code/w2v2_large_512/dev_code'
output_dir = '/home/daniel094144/data/SQA_code/w2v2_large_512/dev_code'

for passage, segment_list in tqdm(segment_dict.items()):
    for idx, id in enumerate(segment_list):
        code_path = data_dir + '/' + 'context-' + passage + '_' + id + '.code'
        cnt_path = data_dir + '/' + 'context-' + passage + '_' + id + '.cnt'
        code = np.loadtxt(code_path)
        cnt = np.loadtxt(cnt_path)
        if idx == 0:
            merge_passage = code
            merge_cnt = cnt
        else: 
            try:
                merge_passage = np.concatenate([merge_passage, code], axis=-1)
                merge_cnt = np.concatenate([merge_cnt, cnt], axis=-1)
            except: 
                print(f'passage: {passage} len {merge_passage.shape[-1]}')
                code = np.array([code])
                cnt = np.array([cnt])
                merge_passage = np.concatenate([merge_passage, code], axis=-1)
                merge_cnt = np.concatenate([merge_cnt, cnt], axis=-1)

    
    output_code = output_dir + '/' + 'context-' + passage + '.code'
    output_cnt = output_dir + '/' + 'context-' + passage + '.cnt'
    np.savetxt(output_code, merge_passage, fmt='%i')    
    np.savetxt(output_cnt, merge_cnt, fmt='%i')



