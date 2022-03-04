import textgrid
import glob 
import os 
from utils import text_preprocess

def find_timespan(tg_file, answer):
    try: 
        tg = textgrid.TextGrid.fromFile(tg_file)
    except: 
        return [0], [0]
    space_idx = [i for i in range(len(tg[0])) if tg[0][i].mark == '']
    words = [text_preprocess(tg[0][i].mark) for i in range(len(tg[0])) if tg[0][i].mark != '']
    
    words_SIL = [text_preprocess(tg[0][i].mark) for i in range(len(tg[0]))]

    pos_map = [i for i, word in enumerate(words_SIL) if word != '']
    answer = str(answer)

    sentence = ' '.join(words)

    # print(sentence)
    match_idxs = [i for i in range(len(sentence)) if sentence.startswith(answer, i)]
    # print(match_idxs)
    if len(match_idxs) == 0: 
        return [0], [0]
    else: 
        start_times, end_times = [], []
        for match_idx in match_idxs:
            match_span = sentence[match_idx:match_idx + len(answer)]

            span = match_span.split()
            
            res = [words[idx:idx + len(span)] == span for idx in range(len(words))]
            try:
                index = res.index(True)
            except: 
                return [0], [0]
            start_times.append(tg[0][pos_map[index]].minTime)
            end_times.append(tg[0][pos_map[index + len(span) - 1]].maxTime)

    
        return start_times, end_times

s, e = find_timespan('/home/daniel094144/Daniel/force_align/force_correct_dev/context-15_31.TextGrid', 'accustomed union and the principle of non discrimination')
