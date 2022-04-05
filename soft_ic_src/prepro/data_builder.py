import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
from tqdm import tqdm
import torch
from multiprocess import Pool
import numpy as np
from others.logging import logger
from others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer
import nltk
from others.utils import clean
from prepro.utils import _get_word_ngrams
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('/tf/project/stanford-corenlp-4.2.2/')

def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)

"""
使用clean_ic里的数据，clean_ic里的json数据包含：hash_code, src, tgt ,image_hash,注意image_hash是没有pad的数据
"""

def load_json(p, lower):
#     source = []
#     tgt = []
#     flag = False
#     for sent in json.load(open(p))['sentences']:
#         tokens = [t['word'] for t in sent['tokens']]
    source = []
    tgt = []
    flag = False
    f = open(p,'r')
    sen_list =[]
    hash_code = p.split('/')[-1].split('.')[0]
    image_hash = []
    for lines in f.readlines():

        if lines=='':
            continue
        try:
            lines.replace('%',' ')
            sen_list.append(nltk.word_tokenize(lines.strip()))
        except:
            print("error",lines)
    f.close()
    for tokens in sen_list:
        if len(tokens)<1:
            continue
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight') or (tokens[0]=='@' and tokens[1] == 'highlight'):
            flag = True
            tgt.append([])
            if tokens[0]=='@highlight':
                image_hash.append(tokens[1])
            else:
                image_hash.append(tokens[2])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    
    return hash_code, source, tgt, image_hash


def load_json_test(p, lower):
    #     source = []
    #     tgt = []
    #     flag = False
    #     for sent in json.load(open(p))['sentences']:
    #         tokens = [t['word'] for t in sent['tokens']]
    source = []
    tgt = []
    flag = False
    f = open(p, 'r')
    sen_list = []
    hash_code = p.split('/')[-1].split('.')[0]
    image_hash = []
    for lines in f.readlines():

        if lines == '':
            continue
        try:
            lines.replace('%', ' ')
            sen_list.append(nltk.word_tokenize(lines.strip()))
        except:
            print("error", lines)
    f.close()
    count = 0
    for tokens in sen_list:
        if len(tokens) < 1:
            continue
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight') or (tokens[0] == '@' and tokens[1] == 'highlight'):
            flag = True

            tgt.append([])
            if tokens[0] == '@highlight':
                image_hash.append(tokens[1])
            else:
                image_hash.append(tokens[2])
            continue
        if (flag):
            tgt[-1].extend(tokens)
            count += 1
            if count>=5:
                break
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]

    return hash_code, source, tgt, image_hash


def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

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


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])

    abstract = _rouge_clean(' '.join(abstract)).split()

    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    rouge = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1

        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)

        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        
        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
         
        src_subtokens = self.tokenizer.tokenize(text)
        
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
      
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
         
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
         
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        return b_data_dict
    
    def preprocess_sentlabel(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        
        _sent_labels = [[0] * len(src) for i in range(len(sent_labels))]
        
        for l in range(len(sent_labels)):
            _sent_labels[l][sent_labels[l]] = 1
        
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
#         sent_labels = [_sent_labels[i] for i in idxs]
        sent_labels = [[_sent_labels[s][i] for i in idxs] for s in range(len(_sent_labels))]
        src = src[:self.args.max_src_nsents]
        sent_labels = [sent_labels[i][:self.args.max_src_nsents] for i in range(len(sent_labels))]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
         
        src_subtokens = self.tokenizer.tokenize(text)
        
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
      
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
         
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
         
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
#         sent_labels = sent_labels[:len(cls_ids)]
        sent_labels = [sent_labels[i][:len(cls_ids)] for i in range(len(sent_labels))]
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        return b_data_dict
    

def format_to_bert(args):
    print(args.dataset)
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        if corpus_type in ['train', 'valid']:continue

        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
                real_name = json_f.split('/')[-1]
                a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert_new, a_lst):
            pass

        pool.close()
        pool.join()
MAX_IMAGE_NUM=5
def get_image_data(image_hash,corpus_type):
    pth ='/tf/dataset/MSMO/'+corpus_type+'_data/'+'img/'+ image_hash+'.jpg'
    transform1 = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor()  # [7]
    ])
    transform2 = transforms.Compose([  # [1
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])
    _image = Image.open(pth)
    _image = transform1(_image)
    if _image.shape[0] != 3:
        print('1 dim image')
        _image = _image.repeat(3, 1, 1)
    _image = transform2(_image)
    """
    _image.shape = 3,244,224
    """
    return _image.numpy()
def _format_to_bert(params):
    corpus_type, json_file, args, save_file= params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    
    bert = BertData(args)
    
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
   
    datasets = []
    for d in jobs:
        source, tgt,image_hash ,hash_code = d['src'], d['tgt'], d['image_hash'], d['hash_code']
        sent_labels = []

        new_tgt=[]
        temp_image_hash=[]
 
        if len(tgt)>=MAX_IMAGE_NUM:
            len_index = MAX_IMAGE_NUM
        else:
            len_index =len(tgt)
        for i in range(len_index):
            temp_out = greedy_selection(source[:args.max_src_nsents], [tgt[i]], 1)
            if temp_out==[]:
                pass
            else:
                sent_labels+=temp_out
                # temp_image_data.append(get_image_data(image_hash=image_hash[i],corpus_type=corpus_type))
                temp_image_hash.append(image_hash[i])
                new_tgt.append(tgt[i])

        if len(new_tgt)<MAX_IMAGE_NUM and (not new_tgt==[]):
            sent_labels+=[sent_labels[-1]]*(MAX_IMAGE_NUM-len(new_tgt))
            temp_image_hash += [temp_image_hash[-1]]*(MAX_IMAGE_NUM-len(new_tgt))
            new_tgt += [new_tgt[-1]] * (MAX_IMAGE_NUM - len(new_tgt))
        elif new_tgt==[]:
            print(tgt)
            print(source)
            print('error, len_tgt',len(new_tgt))
            continue

        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
#         b_data = bert.preprocess(source, new_tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
#                                  is_test=is_test)

        b_data = bert.preprocess(source, new_tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
        
        if (b_data is None):
            continue


        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt,'hash_code':hash_code,'image_hash':temp_image_hash}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert_new(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    for d in jobs:
        source, tgt, image_hash, hash_code = d['src'], d['tgt'], d['image_hash'], d['hash_code']
        sent_labels = []
        new_tgt = []
        temp_image_hash = []
        len_index = len(tgt)
        for i in range(len_index):
            temp_out = greedy_selection(source[:args.max_src_nsents], [tgt[i]], 1)
            if temp_out == []:
                pass
            else:
                sent_labels += temp_out
                temp_image_hash.append(image_hash[i])
                new_tgt.append(tgt[i])
        if new_tgt==[] or temp_image_hash ==[]:
            print(source)
            continue
        b_data_dict = bert.preprocess(source, new_tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)

        if (b_data_dict is None):
            continue
        b_data_dict['hash_code'] = hash_code
        b_data_dict['image_hash'] = temp_image_hash
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

def _format_to_bert_nopad(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    try:
        jobs = json.load(open(json_file))
    except:
        return
    datasets = []
    counts = 0
    for d in jobs:
        source, tgt, image_hash, hash_code = d['src'], d['tgt'], d['image_hash'], d['hash_code']
        sent_labels = []
        new_tgt = []
        temp_image_hash = []
        rouge =[]
        image_hash_dict= {}
        source= [x for x in source if len(x) > args.min_src_ntokens_per_sent]
        for i in range(len(tgt)):
            temp_out ,rouge_score= greedy_selection(source[:args.max_src_nsents], [tgt[i]], 1)
            if temp_out == []:
                print(tgt)
                print(tgt[i])
                print('------------',image_hash[i])
                pass
            else:
                sent_labels += temp_out
                rouge+=rouge_score
                # temp_image_data.append(get_image_data(image_hash=image_hash[i],corpus_type=corpus_type))
                temp_image_hash.append(image_hash[i])
                image_hash_dict[image_hash[i]] = temp_out
                new_tgt.append(tgt[i])
        max_id, max_rouge = -1, 0.0
        if rouge==[]:
            print('error')
            print(sent_labels)
            print(new_tgt)
            counts+=1
            continue
        for i,_socre in enumerate(rouge):
            if _socre>max_rouge:
                max_rouge = _socre
                max_id = i
        image_label = [0 for i in range(len(rouge))]
        image_label[max_id] = 1


        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            new_tgt = [' '.join(s).lower().split() for s in new_tgt]
        #         b_data = bert.preprocess(source, new_tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
        #                                  is_test=is_test)

        b_data_dict = bert.preprocess(source, new_tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data_dict is None):
            continue

        b_data_dict['hash_code'] = hash_code
        b_data_dict['image_hash'] = temp_image_hash
        b_data_dict['iamge_label'] = image_label
        b_data_dict['rouge'] = rouge
        b_data_dict['label_dict'] = image_hash_dict
        datasets.append(b_data_dict)
    print('fasle num:',counts)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
#     corpus_mapping = {}
#     for corpus_type in ['valid', 'test', 'train']:
#         temp = []
#         for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
#             temp.append(hashhex(line.strip()))
#         corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
#         print(corpus_mapping[corpus_type])
#     train_raw_list ={i.split('.')[0]:1 for i in os.listdir('/tf/dataset/MSMO/Presumm_data/train_set/')}
#     valid_raw_list ={i.split('.')[0]:1 for i in os.listdir('/tf/dataset/MSMO/Presumm_data/valid_set/')}
#     test_raw_list = {i.split('.')[0]:1 for i in os.listdir('/tf/dataset/MSMO/Presumm_data/test_set/')}
#     corpus_mapping['train'] = train_raw_list
#     corpus_mapping['valid'] = valid_raw_list
#     corpus_mapping['test'] = test_raw_list
#     train_files, valid_files, test_files = [], [], []
    
#     for f in glob.glob(pjoin(args.raw_path, '*.json')):
#         real_name = f.split('/')[-1].split('.')[0]
#         if (real_name in corpus_mapping['valid']):
#             valid_files.append(f)
#         elif (real_name in corpus_mapping['test']):
#             test_files.append(f)
#         elif (real_name in corpus_mapping['train']):
#             train_files.append(f)
        # else:
        #     train_files.append(f)
    train_files,valid_files,test_files=[],[],[]
    for i in os.listdir('/tf/project/MSMO_presum/MSMO/soft_data/train_data/'):
        train_files.append('/tf/project/MSMO_presum/MSMO/soft_data/train_data/'+i)
    for i in os.listdir('/tf/project/MSMO_presum/MSMO/soft_data/valid_data/'):
        valid_files.append('/tf/project/MSMO_presum/MSMO/soft_data/valid_data/'+i)
    for i in os.listdir('/tf/project/MSMO_presum/MSMO/new_ic_datset/test_data/'):
        test_files.append('/tf/project/MSMO_presum/MSMO/new_ic_datset/test_data/'+i)
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        if corpus_type in ['valid', 'train']:continue
        is_test = False
        if corpus_type == 'test':
            is_test = True
        a_lst = [(f, args,is_test) for f in corpora[corpus_type]]

        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0

        for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst)):
            if len(d['src'])<=len(d['tgt']):
                continue
            dataset.append(d)
            print(len(dataset),args.shard_size)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                print(pt_file+'----------------------------------------------')
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args,is_test = params
#     print(f)
    if not is_test:
        hash_code, source, tgt, image_hash = load_json(f, args.lower)
    else:
        hash_code, source, tgt, image_hash = load_json_test(f, args.lower)
    return {'hash_code':hash_code,'src': source, 'tgt': tgt,'image_hash':image_hash}




def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames
        
        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
