# +
import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os 
import json 

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax
from utils import aggregate_dev_result, AOS_scores, Frame_F1_scores

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/daniel094144/data_folder/SQA_code', type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--output_dir', default='./', type=str)
parser.add_argument('--output_fname', default=None, type=str)
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
model = LongformerForQuestionAnswering.from_pretrained(args.model_path).cuda()
model.eval()


'''
post-processing the answer prediction
'''
def _get_best_indexes(probs, context_offset, n_best_size):
    # use torch for faster inference
    # do not need to consider indexes for question
    best_indexes = torch.topk(probs[context_offset:],n_best_size).indices + context_offset
    return best_indexes
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes
    

def post_process_prediction(start_prob, end_prob,context_offset, n_best_size=10, max_answer_length=500, weight=0.6):
    prelim_predictions = []
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()
    input_id = input_id.squeeze()
    
    start_indexes = _get_best_indexes(start_prob,context_offset, n_best_size)
    end_indexes = _get_best_indexes(end_prob,context_offset, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions. This is taken care in _get_best_indexes
            # if start_index >= len(input_id):
            #     continue
            # if end_index >= len(input_id):
            #     continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            predict = {
                        'start_prob': start_prob[start_index],
                        'end_prob': end_prob[end_index],
                        'start_idx': start_index, 
                        'end_idx': end_index, 
                      }

            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions, 
                                key=lambda x: ((1-weight)*x['start_prob'] + weight*x['end_prob']), 
                                reverse=True)
    if len(prelim_predictions) > 0:
        final_start_idx = prelim_predictions[0]['start_idx']
        final_end_idx = prelim_predictions[0]['end_idx']
    else:
        final_start_idx = torch.argmax(start_prob).cpu()
        final_end_idx = torch.argmax(end_prob).cpu()
    return final_start_idx, final_end_idx



class SQADevDataset(Dataset):
    def __init__(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'dev_code_answer.csv'))
        with open(os.path.join(data_dir, 'dev-hash2question.json')) as f:
            h2q = json.load(f)      
            
        df['question'] = df['hash'].apply(lambda x: h2q[x])
        
        code_dir = os.path.join(data_dir, 'dev-hubert-128-22')
        code_passage_dir = os.path.join(data_dir, 'dev-hubert-128-22')
        context_id = df['context_id'].values
        question = df['question'].values
        code_start = df['code_start'].values
        code_end = df['code_end'].values
        
        self.encodings = []
        for context_id, question_id, start_idx, end_idx in tqdm(zip(context_id, question, code_start, code_end)):
            context = np.loadtxt(os.path.join(code_passage_dir, 'context-'+context_id+'.code')).astype(int)
            question = np.loadtxt(os.path.join(code_dir, question_id+'.code')).astype(int)
            context_cnt = np.loadtxt(os.path.join(code_passage_dir, 'context-'+context_id+'.cnt')).astype(int)
            # question_cnt = np.loadtxt(os.path.join(code_dir, question_id+'.cnt')).astype(int)
            # 0~4 index is the special token, so start from index 5
            # the size of discrete token is 128, indexing from 5~132
            context += 5
            question += 5

            '''
            <s> question</s></s> context</s>
            ---------------------------------
            <s>: 0
            </s>: 2

            '''
            tot_len = len(question) + len(context) + 4 
            
            start_positions = 1 + len(question) + 1 + 1 + start_idx
            end_positions = 1 + len(question) + 1 + 1 + end_idx
            if end_positions > 4096:
                print('end position: ', end_positions)
                start_positions, end_positions = 0, 0
                code_pair = [0]+list(question)+[2]+[2]+list(context)
                code_pair = code_pair[:4095] + [2]
            
            elif tot_len > 4096 and end_positions <= 4096:
                print('length longer than 4096: ', tot_len)
                code_pair = [0]+list(question)+[2]+[2]+list(context)
                code_pair = code_pair[:4095] + [2]
            else:
                code_pair = [0]+list(question)+[2]+[2]+list(context)+[2]
            

            encoding = {}

            encoding.update({'input_ids': torch.LongTensor(code_pair), 
                              'start_positions': start_positions,
                              'end_positions': end_positions,
                              'context_begin': len(question) + 3,  # [0] [2] [2]
                              'context_cnt': context_cnt,
                              })
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]
        

def collate_dev_fn(batch):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    # padding
    for example in batch:
        if len(example['input_ids']) > 4096:
            print('too long:', len(example['input_ids']))
    input_ids = pad_sequence([example['input_ids'] for example in batch], batch_first=True, padding_value=1) 
    attention_mask = pad_sequence([torch.ones(len(example['input_ids'])) for example in batch], batch_first=True, padding_value=0) 
    start_positions = torch.stack([torch.tensor(example['start_positions'], dtype=torch.long) for example in batch])
    end_positions = torch.stack([torch.tensor(example['end_positions'], dtype=torch.long) for example in batch])
    context_begin = torch.stack([torch.tensor(example['context_begin'], dtype=torch.long) for example in batch])
    context_cnt = pad_sequence([torch.tensor(example['context_cnt']) for example in batch], batch_first=True, padding_value=0)  

    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'attention_mask': attention_mask, 
        'context_begin': context_begin, 
        'context_cnt': context_cnt,
    }


def idx2sec(pred_start_idx, pred_end_idx, context_begin, context_cnt):
    context_cnt = context_cnt.squeeze()
    start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_start_idx - context_begin].size()), context_cnt[:pred_start_idx - context_begin])
    end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_end_idx - context_begin].size()), context_cnt[:pred_end_idx - context_begin])
    start_idx, end_idx = torch.sum(start_frame_idx), torch.sum(end_frame_idx)

    return float(start_idx*0.02), float(end_idx*0.02)

##############

#TODO: read all the document in inference

batch_size = 4
valid_dataset = SQADevDataset(data_dir)
dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_dev_fn, shuffle=False)


df = pd.read_csv(os.path.join(data_dir, 'dev_code_answer.csv'))

with open(os.path.join(data_dir, 'dev-hash2question.json'), 'r') as f:
    h2q = json.load(f)
df['question'] = df['hash'].apply(lambda x: h2q[x])
# different answer annotators 
dup = df.duplicated(subset=['hash'], keep='last').values
start_secs = df['new_start'].values
end_secs = df['new_end'].values


f1s_before = []
f1s_after = []
f1s_after_sec = []
pred_starts = []
pred_ends = []
AOSs = []
with torch.no_grad():
    i = 0
    for batch in tqdm(dataloader):
        outputs = model(input_ids=batch['input_ids'].cuda(),
                                      attention_mask=batch['attention_mask'].cuda())
        # start_logits: (B, seq_len)
        pred_start = torch.argmax(outputs.start_logits, dim=1)
        pred_end = torch.argmax(outputs.end_logits, dim=1)
        
        start_prob = softmax(outputs.start_logits, dim=1)
        end_prob = softmax(outputs.end_logits, dim=1)

        logsoftmax = LogSoftmax(dim=1)
        start_logprob = logsoftmax(outputs.start_logits)
        end_logprob = logsoftmax(outputs.end_logits)
    
        final_starts, final_ends = [], [] 
        if batch_size == 1: 
            final_starts, final_ends = post_process_prediction(start_logprob, end_logprob, 
                                                         batch['context_begin'], 3, 275)
        
        else: 
            for j in range(start_logprob.shape[0]):
                final_start, final_end = post_process_prediction(start_logprob[j], end_logprob[j], 
                                                             batch['context_begin'][j], 3, 275)
                final_starts.append(final_start)
                final_ends.append(final_end)

        
        final_start_secs, final_end_secs = [], []
        for final_start, final_end, context_begin, context_cnt  in zip(final_starts, final_ends, batch['context_begin'].cpu(), batch['context_cnt'].cpu()): 
            final_start_sec, final_end_sec = idx2sec(final_start, final_end, context_begin, context_cnt)
            final_start_secs.append(final_start_sec)
            final_end_secs.append(final_end_sec)

        f1_after_sec = Frame_F1_scores(start_secs[i:i+batch_size], end_secs[i:i+batch_size], 
                            final_start_secs, final_end_secs)
        AOS_sec = AOS_scores(start_secs[i:i+batch_size], end_secs[i:i+batch_size],
                            final_start_secs, final_end_secs)
        
        print(f1_after_sec, AOS_sec)
        f1s_after_sec += f1_after_sec
        AOSs += AOS_sec
        pred_starts += final_start_secs
        pred_ends += final_end_secs

        i += batch_size

output_df = pd.DataFrame(list(zip(df['question'].values, start_secs, end_secs, pred_starts, pred_ends, f1s_after_sec, AOSs, dup)), 
                columns=['question', 'gt_start', 'gt_end', 'pred_start', 'pred_end', 'f1', 'AOS', 'dup'])

output_df.to_csv(os.path.join(args.output_dir, args.output_fname+'.csv'))

agg_dev_Frame_F1_score_after_sec = aggregate_dev_result(dup, f1s_after_sec)
agg_dev_AOSs = aggregate_dev_result(dup, AOSs)

print(args.output_fname)
print('post-processed f1 sec: ', agg_dev_Frame_F1_score_after_sec)
print('post-processed aos sec: ', agg_dev_AOSs)

with open(args.output_fname+'.txt', 'w') as f:
    f.write(args.output_fname)
    f.write('post-processed f1 sec: ' + str(agg_dev_Frame_F1_score_after_sec))
    f.write('post-processed aos sec: ' + str(agg_dev_AOSs))


