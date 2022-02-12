# +
import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm

data_name = 'hubert_large_23_128'

file_dir = '/home/daniel094144/data_folder/SQA_code/'+data_name+'/train_code'
df = pd.read_csv('/home/daniel094144/Daniel/E2E-SpokenQA/train_ans_reduced.csv')
start = df['new_start'].values
end = df['new_end'].values
passage = df['context_id'].values

code_start = []
code_end = []

for start_sec, end_sec, context_id in tqdm(zip(start, end, passage)):
#     context_code = np.loadtxt(os.path.join(file_dir, context_id+'.code'))
#     context_cnt = np.loadtxt(os.path.join(file_dir, context_id+'.cnt'))
    context_code = np.loadtxt(os.path.join(file_dir, 'context-'+context_id+'.code'))
    context_cnt = np.loadtxt(os.path.join(file_dir, 'context-'+context_id+'.cnt'))
    start_ind = start_sec / 0.02
    end_ind = end_sec / 0.02
    context_cnt_cum = np.cumsum(context_cnt)
    
    new_start_ind, new_end_ind = None, None
    # print(start_ind, end_ind)
    prev = 0
    for idx, cum_idx in enumerate(context_cnt_cum): 
        
        if cum_idx >= start_ind and new_start_ind is None:
            if abs(start_ind - prev) <= abs(cum_idx - start_ind):
                new_start_ind = idx - 1
            else:
                new_start_ind = idx
        if cum_idx >= end_ind and new_end_ind is None:
            if abs(end_ind - prev) <= abs(cum_idx - end_ind):
                new_end_ind = idx - 1
            else:
                new_end_ind = idx
        prev = cum_idx
    if new_start_ind == None: 
        new_start_ind = idx
    if new_end_ind == None: 
        new_end_ind = idx
    
    code_start.append(new_start_ind)
    code_end.append(new_end_ind)
    # print(context_cnt_cum[new_start_ind], context_cnt_cum[new_end_ind])
df['code_start'] = code_start
df['code_end'] = code_end    
    
df.to_csv('/home/daniel094144/data_folder/SQA_code/'+data_name+'/train_code.csv')


file_dir = '/home/daniel094144/data_folder/SQA_code/'+data_name+'/dev_code'
df = pd.read_csv('/home/daniel094144/Daniel/E2E-SpokenQA/dev_ans_reduced.csv')
start = df['new_start'].values
end = df['new_end'].values
passage = df['context_id'].values

code_start = []
code_end = []

for start_sec, end_sec, context_id in tqdm(zip(start, end, passage)):
    context_code = np.loadtxt(os.path.join(file_dir, 'context-'+context_id+'.code'))
    context_cnt = np.loadtxt(os.path.join(file_dir, 'context-'+context_id+'.cnt'))
    start_ind = start_sec / 0.02
    end_ind = end_sec / 0.02
    context_cnt_cum = np.cumsum(context_cnt)
    
    new_start_ind, new_end_ind = None, None
    prev = 0
    for idx, cum_idx in enumerate(context_cnt_cum): 
        
        if cum_idx >= start_ind and new_start_ind is None:
            if abs(start_ind - prev) <= abs(cum_idx - start_ind):
                new_start_ind = idx - 1
            else:
                new_start_ind = idx
        if cum_idx >= end_ind and new_end_ind is None:
            if abs(end_ind - prev) <= abs(cum_idx - end_ind):
                new_end_ind = idx - 1
            else:
                new_end_ind = idx
        prev = cum_idx
    if new_start_ind == None: 
        new_start_ind = idx
    if new_end_ind == None: 
        new_end_ind = idx
    
    code_start.append(new_start_ind)
    code_end.append(new_end_ind)

df['code_start'] = code_start
df['code_end'] = code_end    
    
df.to_csv('/home/daniel094144/data_folder/SQA_code/'+data_name+'/dev_code.csv')
