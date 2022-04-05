import copy
import os
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
import re
from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder,ExtImageTransformerEncoder
from models.optimizers import Optimizer
from torchvision import models
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}
def calculate_pic_precision(ref_path, pred_path):
    pic_dic = {}
    from prepro.utils import _get_word_ngrams
    f = open('/tf/dataset/MSMO/image_annotation.txt', 'r')
    temp_line = [lines.strip().split() for lines in f.readlines()]
    f.close()
    for i in temp_line:
        if i[0] == 'None':
            continue
        hash_code = i[0].split('_')[0]
        pic_dic[hash_code] = [j.split('.')[0] for j in i]

    pred_file = os.listdir(pred_path)
    ref_file = os.listdir(ref_path)

    pred_dict = {}
    for temp_file in pred_file:
        hash_code = temp_file.split('_')[0]
        f = open(pred_path + temp_file, 'r')
        temp_context = ' '.join(f.readlines()).strip()
        f.close()
        if hash_code not in pred_dict.keys():
            pred_dict[hash_code] = [[temp_file.split('.')[0], temp_context]]
        else:
            pred_dict[hash_code].append([temp_file.split('.')[0], temp_context])

    ref_dict = {}
    for temp_file in ref_file:
        hash_code = temp_file.split('.')[0]
        f = open(ref_path + temp_file, 'r')
        temp_context = ' '.join(f.readlines()).strip()
        f.close()
        ref_dict[hash_code] = temp_context

    commen_hash = set(pred_dict.keys()) & set(pic_dic.keys()) & set(ref_dict.keys())

    find_dict = {}

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    for i in commen_hash:
        candidate_pic_list = pred_dict[i]
        max_score = -1
        max_image_hash = ""
        ref_context = _rouge_clean(ref_dict[i]).split()
        reference_1grams = _get_word_ngrams(1, [ref_context])
        reference_2grams = _get_word_ngrams(2, [ref_context])
        for temp_pic in candidate_pic_list:
            pic_hash, pred_context = temp_pic
            pred_context = _rouge_clean(pred_context).split()
            pred_1grams = _get_word_ngrams(1, [pred_context])
            pred_2grams = _get_word_ngrams(2, [pred_context])
            rouge1 = cal_rouge(pred_1grams, reference_1grams)['f']
            rouge2 = cal_rouge(pred_2grams, reference_2grams)['f']

            if max_score<rouge1+rouge2:
                max_score= rouge1+rouge2
                max_image_hash = pic_hash

        find_dict[i] = max_image_hash

    TP,sums=0.0,0.0
    for temp_hash in commen_hash:

        temp_ref_list = pic_dic[temp_hash]
        if find_dict[temp_hash] in temp_ref_list:
            TP+=1.0
        sums+=1.0
    print(TP/sums)
    return TP/sums


calculate_pic_precision(ref_path='/tf/home_project/PreSumm/data/pred_ic/test_pred_ic_msmo/',
                        pred_path='/tf/home_project/PreSumm/data/pred_ic/test_ic_finetue_with_resnet152/')