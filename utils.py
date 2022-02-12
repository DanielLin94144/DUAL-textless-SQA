# -*- coding: utf-8 -*-
# +
import re
from builtins import str as unicode

def text_preprocess(text):
    text = unicode(text)
    
    text = normalize_numbers(text)
    
    text = text.lower()
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace(r"\%", "percent")
    text = re.sub("-", " ", text)
    text = re.sub("[^ a-z]", "", text)

    return text

# from g2p-en/expand.py
import inflect
import re

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


'''
metric calculation
'''
def compare(pred_start, pred_end, gold_start, gold_end):
    if pred_start >= pred_end: 
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif pred_end <= gold_start or pred_start >= gold_end:
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif gold_end == gold_start: 
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True    
    else:
        no_overlap = False
        if pred_start <= gold_start:
            Min = pred_start
            overlap_start = gold_start
        else: 
            Min = gold_start
            overlap_start = pred_start

        if pred_end <= gold_end:
            Max = gold_end
            overlap_end = pred_end
        else: 
            Max = pred_end
            overlap_end = gold_end
        
    return overlap_start, overlap_end, Min, Max, no_overlap

def Frame_F1_scores(pred_starts, pred_ends, gold_starts, gold_ends):
    F1s = []
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
        if no_overlap: 
            if pred_start == gold_start and pred_end == gold_end:
                F1 = 1
            else: 
                F1 = 0
        else: 
            Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
            Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
            F1 = float(2 * Precision * Recall / (Precision + Recall))
        F1s.append(F1)
    return F1s

def Frame_F1_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
    if no_overlap: 
        if pred_start == gold_start and pred_end == gold_end:
            F1 = 1
        else: 
            F1 = 0
    else: 
        Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
        Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1

def AOS_scores(pred_starts, pred_ends, gold_starts, gold_ends):
    AOSs = []
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)

        if no_overlap: 
            AOS = 0
        else: 
            AOS = float((overlap_end - overlap_start) / (Max - Min))
        AOSs.append(AOS)
    return AOSs

def AOS_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)

    if no_overlap: 
        AOS = 0
    else: 
        AOS = (overlap_end - overlap_start) / (Max - Min)
    return AOS
        


def aggregate_dev_result(dup, metric):
    aggregate_result = []
    buff = []
    for i in range(len(dup)):
        if not dup[i]:
            if len(buff) == 0:
                aggregate_result.append(metric[i])
            else: 
                aggregate_result.append(max(buff))

            buff = []  # clear buffer

        else: 
            buff.append(metric[i])
    
    return sum(aggregate_result) / len(aggregate_result)

def calc_overlap(pred_starts, pred_ends, gold_starts, gold_ends):
    x = [pred_starts, pred_ends]
    y = [gold_starts, gold_ends]
    if x[1] <= y[0] or x[0] >= y[1]:
        return 0.0, 0.0
    minest, maxest = min(x[0], y[0]), max(x[1], y[1])
    left, right = max(x[0], y[0]), min(x[1], y[1])
    try:
        aos = (right - left) / (maxest - minest)
        precision = (right - left) / (x[1] - x[0])
        recall = (right - left) / (y[1] - y[0])
        f1 = float((2 * precision * recall) / (precision + recall))
    except:
        print(right, left, maxest, minest)
    return f1, aos
