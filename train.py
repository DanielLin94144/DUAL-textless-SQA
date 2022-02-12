# +
import torch
import nlp
from transformers import LongformerTokenizerFast
import os 
import numpy as np
import json
import pandas as pd
from tqdm import tqdm 
import argparse
import yaml 
import json


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class SQADataset(Dataset):
    def __init__(self, data_dir, mode='train', idx_offset=5):
        df = pd.read_csv(os.path.join(data_dir, mode+'_code.csv'))
        with open(os.path.join(data_dir, mode+'-hash2question.json')) as f:
            h2q = json.load(f)      
            
        df['question'] = df['hash'].apply(lambda x: h2q[x])
        
        code_dir = os.path.join(data_dir, mode+'_code')
        code_passage_dir = os.path.join(data_dir, mode+'_code')
        context_id = df['context_id'].values
        question = df['question'].values
        code_start = df['code_start'].values
        code_end = df['code_end'].values
        
        self.encodings = []
        for context_id, question_id, start_idx, end_idx in tqdm(zip(context_id, question, code_start, code_end), total=len(context_id)):
            context = np.loadtxt(os.path.join(code_passage_dir, 'context-'+context_id+'.code')).astype(int)
            question = np.loadtxt(os.path.join(code_dir, question_id+'.code')).astype(int)
            if context.shape == ():
                context = np.expand_dims(context, axis=-1)
            if question.shape == ():
                question = np.expand_dims(question, axis=-1)
            # 0~4 index is the special token, so start from index 5
            # the size of discrete token is 128, indexing from 5~132
            context += idx_offset
            question += idx_offset

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
                              })
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]
        

def collate_batch(batch):
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

    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'attention_mask': attention_mask
    }


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Max input length for the source text"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))
 
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = LongformerForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # reset input embedding
#     selected_emb = model.longformer.embeddings.word_embeddings.weight[:128+5, :]
#     embedding = torch.nn.Embedding.from_pretrained(selected_emb)
#     model.longformer.set_input_embeddings(embedding)
    # if train from scratch
#     print('train from scratch!')
#     model.init_weights()
    # Get datasets
    print('[INFO]    loading data')
    
    train_dataset = SQADataset(data_dir, mode='train')
    dev_dataset = SQADataset(data_dir, mode='dev')

    print('[INFO]    loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_batch,
    )
    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline.yaml', type=str)
    parser.add_argument('--exp_name', default='test', type=str)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    with open('args.json', 'w') as f:
        json.dump(config['Trainer'], f)

    print('[INFO]    Using config {}'.format(args.config))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = config['data']['data_dir']
    main()
