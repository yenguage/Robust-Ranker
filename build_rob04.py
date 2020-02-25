from functools import reduce
from collections import defaultdict
import numpy as np
import pickle, random, json, time, math, re
from six import iteritems
from isu_tool.ui import pgbar
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

data_dir = 'data/robust04'
max_query_len = 20
max_doc_len = 800
batch_size = 10

def time2str(time):
    h = time // 3600
    m = time % 3600 // 60
    s = time % 60
    ret = '%02dh%02dm%02ds' % (h, m, s)

    return ret

def load_json(path):
    print('Loading [Json|%s] ...' % path, end=' ', flush=True)
    start_time = time.time()
    with open(path, 'r') as fp:
        ret = json.loads(fp.read().strip())
    end_time = time.time()
    print('Finish! [%s]' % time2str(end_time - start_time), flush=True)

    return ret

def load_pickle(path):
    print('Loading [Pickle|%s] ...' % path, end=' ', flush=True)
    start_time = time.time()
    with open(path, 'rb') as fp:
        ret = pickle.load(fp)
    end_time = time.time()
    print('Finish! [%s]' % time2str(end_time - start_time), flush=True)

    return ret

clean = lambda t: re.sub('[,?;*!%^&_+():-\[\]{}]', ' ', t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'", ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('-', ' ').replace('.', '').replace('&hyph;', ' ').replace('&blank;', ' ').strip().lower())

def build(mode, tokenizer):#, idfs):
    for split in range(1, 6):
        data_total = []
        docset = load_pickle('%s/split_%d/rob04_bm25_docset_top1000.%s.s%d.pkl' % (data_dir, split, mode, split))
        dataset = load_pickle('%s/split_%d/rob04_bm25_top1000.%s.s%d.pkl' % (data_dir, split, mode, split))

        for data in pgbar(dataset['queries'], pre='[ %s-%d ]' % (mode.upper(), split)):
            # build query info
            query_t       = tokenizer.tokenize(clean(data['query_text']).lower())[:max_query_len]
            query_ids     = tokenizer.convert_tokens_to_ids(query_t)
            # query_ids_raw = [q for q in query_ids]
            query_len     = len(query_ids)
            # query_idf     = [idfs(t) for t in query_t]

            # zero padding
            while len(query_ids) < max_query_len:
                query_ids.append(0)
                # query_idf.append(0.)

            # remove too short queries
            if query_len < 2:
                continue

            # prepare to build docs info
            docs_ids    = []
            docs_len    = []
            # docs_idf    = []
            # input_ids   = []
            # input_seg   = []
            # input_mask  = []
            rel_1d      = []
            num_docs    = 0
            num_rel     = 0
            # exact_match = []

            for document in data['retrieved_documents']:
                doc_t = tokenizer.tokenize(clean(docset[document['doc_id']]['abstractText']).lower())[:max_doc_len]
                doc_ids = tokenizer.convert_tokens_to_ids(doc_t)
                # doc_ids_raw = [d for d in doc_ids]
                doc_len = len(doc_ids)
                # doc_idf = [idfs(t) for t in doc_t]
                # em = [1. if t in query_t else 0. for t in doc_t]
                while len(doc_ids) < max_doc_len:
                    doc_ids.append(0)
                    # doc_idf.append(0.)
                    # em.append(0.)

                rel1 = 1 if document['is_relevant'] else 0

                # input_ids_t = query_ids + doc_ids
                # input_seg_t = [0 for _ in range(max_query_len)] + [1 for _ in range(doc_len)] + [0 for _ in range(max_doc_len - doc_len)]
                # input_mask_t = [1 for _ in range(query_len)] + [0 for _ in range(max_query_len - query_len)] + [1 for _ in range(doc_len)] + [0 for _ in range(max_doc_len - doc_len)]
                # while len(input_ids_t) < max_query_len + max_doc_len:
                #     input_ids_t.append(0)
                #     input_seg_t.append(0)
                #     input_mask_t.append(0)

                docs_ids.append(doc_ids)
                docs_len.append(doc_len)
                # docs_idf.append(doc_idf)
                # input_ids.append(input_ids_t)
                # input_seg.append(input_seg_t)
                # input_mask.append(input_mask_t)
                rel_1d.append(rel1)
                # exact_match.append(em)

            if data['num_rel_ret'] < 1:
                continue

            with torch.no_grad():
                dsize  = data['num_ret']
                # ids    = torch.tensor(input_ids)
                # seg    = torch.tensor(input_seg)
                # mask   = torch.tensor(input_mask)
                rel_1d = torch.tensor(rel_1d)

                query     = torch.tensor(query_ids).unsqueeze(0).repeat(dsize, 1)
                # query_idf = torch.tensor(query_idf).unsqueeze(0).repeat(dsize, 1)
                query_len = torch.tensor(query_len).repeat(dsize)
                docs      = torch.tensor(docs_ids)
                # docs_idf  = torch.tensor(docs_idf)
                docs_len  = torch.tensor(docs_len)

                data_total.append({
                    'dsize'       : dsize,
                    # 'ids'         : ids,
                    # 'seg'         : seg,
                    # 'mask'        : mask,
                    'query_id'    : data['query_id'],
                    'query'       : query,
                    # 'query_idf'   : query_idf,
                    'query_len'   : query_len,
                    'docs_id'     : [document['doc_id'] for document in data['retrieved_documents']],
                    'docs'        : docs,
                    # 'docs_idf'    : docs_idf,
                    'docs_len'    : docs_len,
                    'rel_1d'      : rel_1d,
                })

        with open('%s/split_%d/prepro.%s.pkl' % (data_dir, split, mode), 'wb') as fp:
            pickle.dump(data_total, fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# idfs_data = load_pickle('data/robust04/IDF.pkl')
# idfs = lambda w: idfs_data[w] if w in idfs_data else 0.

build('dev', tokenizer)#, idfs)
build('test', tokenizer)#, idfs)
build('train', tokenizer)#, idfs)
