import bisect
import gc
import glob
import random
import numpy as np
import torch
from torchvision import transforms
from others.logging import logger
from PIL import Image

class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _pad_image_feature(self, image_hash, max_caption_num=-1, sent_pad_index=0, is_test=False):
        rtn_data = []
        image_mask = []
        pre_image_data = []
        for x in image_hash:
            temp_data = []
            for hash in x:
                if is_test:
                    _image = self._get_image_data(hash, 'test')
                else:
                    _image = self._get_image_data(hash, 'train')
                temp_data.append(_image)
            pre_image_data.append(temp_data)

        image_data = pre_image_data
        for d in image_data:
            pad_image = torch.tensor(np.zeros(np.array(d[sent_pad_index]).shape))
            # pad_image = d[sent_pad_index]
            if len(d) < max_caption_num:
                image_mask.append([True for i in range(len(d))] + [False] * (max_caption_num - len(d)))
                rtn_data.append(torch.cat(d + [pad_image] * (max_caption_num - len(d)),0).unsqueeze(0))
            else:
                rtn_data.append(torch.cat(d[:max_caption_num],0).unsqueeze(0))
                image_mask.append([True for i in range(max_caption_num)])
        return torch.cat(rtn_data,0).float(), image_mask
    def _get_image_data(self,image_hash, corpus_type ):

        pth = '/tf/dataset/MSMO/' + corpus_type + '_data/' + 'img/' + image_hash + '.jpg'
        # pth = '/tf/dataset/MSMO/' + 'train_data/' + 'img/' + image_hash + '.jpg'
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
        _image = Image.open(pth).convert('RGB')
        _image = transform1(_image)
        if _image.shape[0] != 3:
            _image = _image.repeat(3, 1, 1)
        _image = transform2(_image)
        """
        _image.shape = 3,244,224
        """
        return _image.unsqueeze(0)
    def __init__(self, data=None, device=None, is_test=False,max_caption_num=5,sent_pad_index=0):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]
            image_hash = [x[5] for x in data]


            src = torch.tensor(self._pad(data=pre_src, pad_id=0,width=-1))
            tgt = torch.tensor(self._pad(data=pre_tgt, pad_id=0, width=-1))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = ~(src == 0)
            mask_tgt = ~(tgt == 0)

            clss = torch.tensor(self._pad(pre_clss, -1))

            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0
            image_data =[]
            max_len_img =1
            for i in image_hash:
                if max_len_img<len(i):
                    max_len_img = len(i)
            image_mask = []
            for d in image_hash:
                _temp_image_list = []
                for _img_hash in d:
                    _image = self._get_image_data(_img_hash, 'test')
                    _temp_image_list.append(_image)
                temp_mask = [True]*len(_temp_image_list)
                if len(_temp_image_list)<max_len_img:
                    temp_leng = len(_temp_image_list)
                    _temp_image_list += [torch.zeros(_image.shape)]*(max_len_img-temp_leng)
                    temp_mask+=[False]*(max_len_img-temp_leng)
                temp_mask = torch.tensor(temp_mask).unsqueeze(0)
                # print(torch.cat(_temp_image_list,0).unsqueeze(0).shape)
                image_data.append(torch.cat(_temp_image_list, 0).unsqueeze(0))
                image_mask.append(temp_mask)
            image_data = torch.cat(image_data, 0)
            image_mask = torch.cat(image_mask,0)
            hash_code = [x[6] for x in data]
            setattr(self, 'hash_code', hash_code)
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))
            setattr(self, 'image_hash', image_hash)
            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'image_data', image_data.to(device))
            setattr(self, 'image_mask', image_mask.to(device))
            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset
    
    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + corpus_type + '.[0-9]*.pt'))
    
    if pts:
        if (shuffle):
            random.shuffle(pts)
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    """
    统计当前Minbatch单词个数
    """
    
    if (len(new) == 4):
        pass

    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None
        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)
        
        #return temp
#         return DataIterator(args = self.args,
#             dataset=self.cur_dataset,  batch_size=self.batch_size,
#             device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs






    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        hash_code = ex['hash_code']
        image_hash = ex['image_hash']
        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]
        #src为转为数字后的src，src_txr为没有转为数字的src文本，src_sent_labels的长度为src的句子个数，七中的每个元素为0或1，1代表对应的句子为ground truth。segs长度与src长度一致，每个元素值为0或1，每个句子内的元素元素值相同，相邻两个句子的元素值相反。


        if(is_test):
            return src, tgt, segs, clss, src_sent_labels,image_hash,hash_code,src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels,image_hash,hash_code

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
       
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            p_batch = self.batch(p_batch, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
