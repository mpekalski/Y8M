#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate GAP from validate_labels.csv and validate_predictions.csv.
"""

from datetime import datetime
from concurrent import futures
import functools
import numpy as np
import pandas as pd

def predline2series2(s):
    """ parse one line of predictions.csv
    input: 
        s: string, one line of predictions.csv
    output:
        vid : string, vidio ID
        labs: int, labels of video category in descending probability order
        prob: pd.Series, probabilities of labs.
    """        
    ss = s.split(',')
    vid = ss[0]
    kvs = [x.strip() for x in ss[1].split(' ')]
    d = {}
    i = 0
    while i < len(kvs):
        d[int(kvs[i])] = float(kvs[i+1])
        i += 2
    prob = pd.Series(d).sort_values(ascending=False)
    labs = np.array(prob.index)
    return vid, labs, prob

def predline2series(s):
    """ parse one line of predictions.csv
    input: 
        s: string, one line of predictions.csv
    output:
        vid : string, vidio ID
        labs: int, labels of video category in descending probability order
        prob: pd.Series, probabilities of labs.
    """        
    ss = s.split(',')
    vid = ss[0]
    
    kvs = [x.strip() for x in ss[1].split(' ')]
    k = (int(k) for k in kvs[0::2])
    v = (float(v) for v in kvs[1::2])
    
    prob = pd.Series(v, index=k)
    prob = prob.sort_values(ascending=False)
    labs = np.array(prob.index)
    
    return vid, labs, prob

def trueline2series(s):
    """ parse one line of validate_labels.csv """
    ss = s.split(',')
    vid = ss[0]
    labs = [int(x.strip()) for x in ss[1].split(' ')]
    return vid, labs
    
def average_precision(sorted_pred_labels, true_labels):
    delta_recall= 1. / len(true_labels)
    ap = 0.0
    poscount = 0.0
    for i in range(len(sorted_pred_labels)):
        if sorted_pred_labels[i] in true_labels:
            poscount += 1
            ap += poscount / (i+1) 
    return ap * delta_recall

def gap_from_preds(preds, trues):    
    preds = sorted(preds, key=lambda p: p[0])
    trues = sorted(trues, key=lambda p: p[0])
    gap = 0.0
    cnt = 0.0
    for s1, s2 in zip(preds, trues):
        v1, plab, _ = s1
        v2, tlab    = s2
        if (v1 == v2):
            ap = average_precision(plab, tlab)
            gap += ap
            cnt += 1.0
        else:
            print('cnt = {}, vid1 = {}, vid2 = {}'.format(cnt, v1, v2))
            raise
    return gap/cnt

def gap_from_csv(predcsv, truecsv):
    """ Calculate the global average precision. average by example """
    with open(predcsv) as fpred:
        _     = next(fpred)
        preds = [predline2series(x) for x in fpred]
    with open(truecsv) as ftrue:
        trues = [trueline2series(x) for x in ftrue]
    return gap_from_preds(preds, trues)


def add_probs(p1, p2, wgts=None):
    if wgts==None:
        wgts = [0.5, 0.5]            
    pcomb = wgts[0] * p1
    pcomb = pcomb.add(wgts[1] * p2, fill_value=0.0)
    pcomb = pcomb.sort_values(ascending=False)
    pcomb = pcomb[:20]
    return pcomb

def prob2predline(vid, prob):
    x = ['{} {:0.6f}'.format(k, v) for k, v in zip(prob.index, prob)]
    x_str = ' '.join(x)
    res = vid + ',' + x_str
    return res
            

def add_pred_line(ss, wgts=None):
    s1, s2 = ss
    v1, _, p1 = predline2series(s1)
    v2, _, p2 = predline2series(s2)
    
    if v1 != v2:
        raise("Mis-matched video IDs: "+v1+" "+v2)
    pcomb = add_probs(p1, p2, wgts)
    p_str = prob2predline(v1, pcomb)
    return p_str

def add_pred_series(ss, wgts=None):
    s1, s2 = ss
    v1, _, p1 = s1
    v2, _, p2 = s2
    
    if v1 != v2:
        print("Mis-matched video IDs: {} vs {}.".format(v1, v2))
        raise 
    pcomb = add_probs(p1, p2, wgts)
    p_str = prob2predline(v1, pcomb)
    return p_str

def file_2_series(fn):
    with open(fn) as f:
        header = next(f)
        preds = [predline2series(x) for x in f]
    return preds

def sort_by_vid(preds):
    return sorted(preds, key=lambda x: x[0])

def combine_pred_csv(fn1, fn2, fn_out='/tmp/combo.csv', wgts=None):
    """ linear add the probabilities from two prediction.csv files.
    inputs:
        fn1, fn2: files to be combined.
        fn_out: output file name
        wgts: a list of two values, for example, [0.5, 0.5]
    output:
        no return values
    """
    
    executor = futures.ProcessPoolExecutor(max_workers=2)

    t1 = datetime.now()
    print('start combination at ', t1)
     
    preds1, preds2 = executor.map(file_2_series, (fn1, fn2))       
   
    t2 = datetime.now()
    print('files read by', t2)
    
    return combine_preds_2_csv(preds1, preds2, fn_out, wgts)

def combine_preds_2_csv(preds1, preds2, fn_out='/tmp/combo.csv', wgts=None):
    t1 = datetime.now()
    print('start combination at ', t1)
    executor = futures.ProcessPoolExecutor(max_workers=6)
    add_pred_series_wgts = functools.partial(add_pred_series, wgts=wgts)
    
    preds1 = sorted(preds1, key=lambda x: x[0])
    preds2 = sorted(preds2, key=lambda x: x[0])
    t2 = datetime.now()
    print('sorted preds2 at ', t2)
        
    lines = executor.map(add_pred_series_wgts, zip(preds1, preds2))
    
    t2 = datetime.now()
    print('finished adding lines at ', t2)
    #print('Lines processed: {}'.format(len(lines)))
    
    cnt = 0             
    with open(fn_out, 'w') as fout:
        fout.write('VideoId,LabelConfidencePairs\n')
        for line in lines:
            fout.write(line+'\n')
            cnt += 1
            
    print('{} prediction lines were written to {}'.format(cnt, fn_out))
    t3 = datetime.now()
    print('finished combination at', t3)
    print('Total run time: {}'.format(t3 - t1))
    return None

def combine_pred_csv_single_thread(fn1, fn2, fn_out='/tmp/combo.csv', wgts=None):
    """ linear add the probabilities from two prediction.csv files.
    inputs:
        fn1, fn2: files to be combined.
        fn_out: output file name
        wgts: a list of two values, for example, [0.5, 0.5]
    output:
        no return values
    """

    t1 = datetime.now()
    print('start combination at ', t1)
     
    preds1 = file_2_series(fn1)
    preds2 = file_2_series(fn2)
   
    t2 = datetime.now()
    print('files read by', t2)
    
    preds1 = sorted(preds1, key=lambda x: x[0])
    preds2 = sorted(preds2, key=lambda x: x[0])
    t2 = datetime.now()
    print('sorted at ', t2)
          
    lines = []
    for ss in zip(preds1, preds2):
        lines.append( add_pred_series(ss, wgts))
    
    t2 = datetime.now()
    print('finished adding lines at ', t2)   
    
    cnt = 0             
    with open(fn_out, 'w') as fout:
        fout.write('VideoId,LabelConfidencePairs\n')
        for line in lines:
            fout.write(line+'\n')
            cnt += 1
            
    print('{} prediction lines were written to {}'.format(cnt, fn_out))
    t3 = datetime.now()
    print('finished combination at', t3)
    print('Total run time: {}'.format(t3 - t1))
    return None
            
                 
