import io
import json
import sys
from os.path import isfile, isdir, join, dirname, abspath, exists
from os import listdir

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from utils.config_loader import path_parser, logger, test_year

from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer
import string  

from tqdm import tqdm
from collections import Counter
import re


class DatasetParser:
    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.uid2instc = {}
    
    def remove_extra_spaces(self, sent):
        sent = re.sub(r'\s+', ' ', sent).strip()
        return sent
    
    def _proc_sent(self, sent, rm_stop, stem, rm_short=None, min_nw_sent=3):
        sent = sent.lower()
        sent = self.remove_extra_spaces(sent)

        if sent == '':
            return ''

        if rm_short and len(word_tokenize(sent)) < min_nw_sent:
            return ''

        if rm_stop:
            sent = remove_stopwords(sent)

        if stem:
            sent = self.porter_stemmer.stem_sentence(sent)

        return sent

    def cid2sents(self, cid, rm_stop=True, rm_dialog=False, stem=True, max_ns_doc=None):
        """
            Load all sentences in a cluster.

        :param cid:
        :param rm_stop:
        :param stem:
        """

        assert self.uid2instc  # have to load uid2instc first

        original_sents, processed_sents = [], []
        for sent in self.uid2instc[cid]['d']:
            proc = self._proc_sent(sent, rm_stop=rm_stop, stem=stem)
            if proc:
                original_sents.append([sent])  # each sentence is treated as a document (for idf)
                processed_sents.append([proc])  # each sentence is treated as a document (for idf)
            
        if max_ns_doc:
            original_sents = original_sents[:max_ns_doc]
            processed_sents = processed_sents[:max_ns_doc]
        assert len(processed_sents) == len(original_sents)
        return original_sents, processed_sents


class WikicatParser(DatasetParser):
    def __init__(self, dataset_var):
        """
            dataset_var: company, animal, film
        """
        DatasetParser.__init__(self)
        self.dataset_var = dataset_var
        self.n_clusters, self.n_paras = 0, 0
        self.uid2instc, self.uids = self.load()
    
    def proc_doc_line(self, line):
        line = line.strip('\n')
        content = line.split(' <EOT> ')[1]
        paras = content.split(' <EOP> ')
        doc = ' '.join(paras)

        self.n_clusters += 1
        self.n_paras += len(paras)

        return doc

    def proc_summ_line(self, line):
        return line.strip('\n')
        
    def load(self):
        doc_file = path_parser.wikicat / f'{self.dataset_var}.source'
        summary_file = path_parser.wikicat / f'{self.dataset_var}.target'

        doc_lines = io.open(doc_file, encoding='utf-8').readlines()
        summ_lines = io.open(summary_file, encoding='utf-8').readlines()
        
        data = {}
        uids = []
        for i in range(len(doc_lines)):
            uid, d, s = str(i), self.proc_doc_line(doc_lines[i]), self.proc_summ_line(summ_lines[i])
            # if i in (101, 232):
            #     print(uid)
            #     print(d)
            #     print(s)
            
            uids.append(uid)
            data[uid] = {
                'd': sent_tokenize(d),
                's': s,
            }
        return data, uids


class WikirefParser(DatasetParser):
    def __init__(self, dataset_var, query_sep='*=-', doc_sent_sep=' ', query_sep_space=True):
        DatasetParser.__init__(self)
        self.dataset_var = dataset_var
        self.query_sep = query_sep

        if query_sep_space:  # add spaces at start and end to query_sep
            self.query_sep = ' ' + self.query_sep + ' '

        self.doc_sent_sep = doc_sent_sep        
        self.uid2instc, self.uids = self.load()

    def load(self):
        data_file = path_parser.wikiref / f'{self.dataset_var}.json'
        with io.open(data_file, encoding='utf-8') as f:
            json_objs = json.load(f)
        
        data = {}
        uids = []
        for json_obj in json_objs:
            uid, d, q, s = json_obj['uid'], json_obj['document'], json_obj['query'], json_obj['summary']
            uids.append(uid)
            data[uid] = {
                'd': d,
                'q': self.query_sep.join(q),
                's': s,
            }
        # data = []
        # for json_obj in json_objs:
        #     uid, d, q, s = json_obj['uid'], json_obj['document'], json_obj['query'], json_obj['summary']
        #     data.append({
        #         'uid': uid,
        #         'd': d,
        #         'q': self.query_sep.join(q),
        #         's': s,
        #     })
        return data, uids


class DebatepediaParser(DatasetParser):
    def __init__(self, dataset_var):
        DatasetParser.__init__(self)
        self.dataset_var = dataset_var
        self.uid2instc, self.uids = self.load()
        
    def proc_line(self, line):
        line = line.rstrip('\n')
        line = line[4:]
        line = line[:-6]
        return line

    def load(self):
        file_d = path_parser.debatepedia / f'{self.dataset_var}_content'
        file_q = path_parser.debatepedia / f'{self.dataset_var}_query'
        file_s = path_parser.debatepedia / f'{self.dataset_var}_summary'
        ds = io.open(file_d, encoding='utf-8').readlines()
        qs = io.open(file_q, encoding='utf-8').readlines()
        ss = io.open(file_s, encoding='utf-8').readlines()
        
        data = {}
        uids = []
        for i in range(len(ds)):
            uids.append(str(i))
            data[str(i)] = {
                # 'd': self.proc_line(ds[i]), 
                'd': sent_tokenize(self.proc_line(ds[i])), 
                'q': self.proc_line(qs[i]), 
                's': self.proc_line(ss[i])
            }

        # data = []
        # for i in range(len(ds)):
        #     data.append({
        #         'uid': str(i),
        #         'd': sent_tokenize(self.proc_line(ds[i])), 
        #         'q': self.proc_line(qs[i]), 
        #         's': self.proc_line(ss[i])
        #     })
        return data, uids


class DUCDocParser(DatasetParser):
    def __init__(self, dataset_var):
        DatasetParser.__init__(self)
        self.dataset_var = dataset_var
        self.year = dataset_var
        self.uid2instc, self.uids = self.load()
        
    def proc_line(self, line):
        line = line.rstrip('\n')
        return line

    def load(self):
        root_dp = path_parser.duc / f'duc_{self.year}.ranked.lines'
        
        file_d = root_dp / f'duc_{self.year}.source'
        file_q = root_dp / f'duc_{self.year}.query'
        # file_meta = root_dp / f'{self.dataset_var}.meta'
        ds = io.open(file_d, encoding='utf-8').readlines()
        qs = io.open(file_q, encoding='utf-8').readlines()
        # ss = io.open(file_meta, encoding='utf-8').readlines()
        
        data = {}
        uids = []
        for i in range(len(ds)):
            uids.append(str(i))
            data[str(i)] = {
                'd': sent_tokenize(self.proc_line(ds[i])), 
                'q': self.proc_line(qs[i]), 
            }
        
        return data, uids


class DUCClusterParser(DatasetParser):
    def __init__(self, dataset_var):
        DatasetParser.__init__(self)
        self.dataset_var = dataset_var
        self.year = dataset_var
        self.uid2instc, self.uids = self.load()
        
    def proc_line(self, line):
        line = line.rstrip('\n')
        return line

    def get_summary(self, cid):
        """
            Find summaries for cid.
        """
        summaries = []
        for fn in listdir(path_parser.duc_summary / self.year):
            if not fn.startswith(cid):
                continue

            lines = io.open(path_parser.duc_summary / self.year / fn, encoding='latin1').readlines()
            summaries.append([self.proc_line(line) for line in lines])
        
        return summaries

    def load(self):
        root_dp = path_parser.duc / f'duc_{self.year}.ranked'
        data = {}
        uids = []
        
        with io.open(root_dp, encoding='utf-8') as f:
            for line in f:
                cc_obj = json.loads(line.strip('\n'))
                cid = cc_obj['cid']
                uids.append(cid)
                docs = [doc["original_doc"] for doc in cc_obj['docs']]
                summary = self.get_summary(cc_obj['cid'])

                query = cc_obj['query']['original_concat']

                data[cid] = {
                    'd': docs, 
                    's': summary, 
                    'q': query,
                }
        
        return data, uids


class TDQFSDocParser(DatasetParser):
    def __init__(self):
        DatasetParser.__init__(self)
        self.uid2instc, self.uids = self.load()
        
    def proc_line(self, line):
        line = line.rstrip('\n')
        return line

    def load(self):
        root_dp = path_parser.tdqfs / f'tdqfs.ranked.lines'
        
        file_d = root_dp / f'tdqfs.source'
        file_q = root_dp / f'tdqfs.query'
        ds = io.open(file_d, encoding='utf-8').readlines()
        qs = io.open(file_q, encoding='utf-8').readlines()
        
        data = {}
        uids = []
        for i in range(len(ds)):
            uids.append(str(i))
            data[str(i)] = {
                'd': sent_tokenize(self.proc_line(ds[i])), 
                'q': self.proc_line(qs[i]), 
            }
        
        return data, uids


class TDQFSClusterParser(DatasetParser):
    def __init__(self):
        DatasetParser.__init__(self)
        self.uid2instc, self.uids = self.load()
        
    def proc_line(self, line):
        line = line.rstrip('\n')
        return line

    def get_summary(self, cid):
        """
            Find summaries for cid.
        """
        summaries = []
        for fn in listdir(path_parser.tdqfs_summary_targets):
            if not fn.startswith(cid):
                continue

            lines = io.open(path_parser.tdqfs_summary_targets / fn).readlines()
            summaries.append([self.proc_line(line) for line in lines])
        
        return summaries

    def load(self):
        root_dp = path_parser.tdqfs / f'tdqfs.ranked'
        data = {}
        uids = []
        
        with io.open(root_dp, encoding='utf-8') as f:
            for line in f:
                cc_obj = json.loads(line.strip('\n'))
                cid = cc_obj['cid']
                uids.append(cid)
                docs = [doc["original_doc"] for doc in cc_obj['docs']]
                summary = self.get_summary(cc_obj['cid'])

                query = cc_obj['query']['title']

                data[cid] = {
                    'd': docs, 
                    's': summary, 
                    'q': query,
                }
        
        return data, uids


if test_year.startswith('debatepedia'):
    dataset_root = path_parser.debatepedia
    dataset_var = test_year.split('_')[-1]
    # dataset_parser = DebatepediaParser(dataset_var=dataset_var)
elif test_year.startswith('wikicat'):
    dataset_root = path_parser.wikicat
    dataset_var = test_year.split('_')[-1]
    dataset_parser = WikicatParser(dataset_var=dataset_var)
elif test_year.startswith('wikiref'):
    dataset_root = path_parser.wikiref
    dataset_var = test_year.split('_')[-1]
    dataset_parser = WikirefParser(dataset_var=dataset_var)
elif test_year.startswith('ducdoc'):
    dataset_root = path_parser.duc
    dataset_var = test_year.split('_')[-1]
    dataset_parser = DUCDocParser(dataset_var=dataset_var)
elif test_year.startswith('duccluster'):
    dataset_root = path_parser.duc
    dataset_var = test_year.split('_')[-1]
    dataset_parser = DUCClusterParser(dataset_var=dataset_var)
elif test_year.startswith('tdqfsdoc'):
    dataset_root = path_parser.tdqfs
    dataset_parser = TDQFSDocParser()
elif test_year.startswith('tdqfscluster'):
    dataset_root = path_parser.tdqfs
    dataset_parser = TDQFSClusterParser()
else:
    raise NotImplementedError


def build_src(max_ns_doc=None):
    print('building source...')
    lines = []
    for uid in tqdm(dataset_parser.uids):
        original_sents, _ = dataset_parser.cid2sents(uid, max_ns_doc=max_ns_doc)
        line = ' '.join([sent[0] for sent in original_sents])  # take each sentence out from its own list
        lines.append(line)
        
    src_fp = dataset_root / f'{dataset_var}.source'
    assert not exists(src_fp), f'remove src file first: {src_fp}'
    io.open(src_fp, 'a').write('\n'.join(lines))


def build_tgt():
    print('building target...')
    lines = []
    for uid in tqdm(dataset_parser.uids):
        summary = dataset_parser.uid2instc[uid]['s']
        if isinstance(summary, list):
            summary = ' '.join(summary)
        lines.append(summary)
        
    tgt_fp = dataset_root / f'{dataset_var}.target'
    assert not exists(tgt_fp), f'remove tgt file first: {tgt_fp}'
    io.open(tgt_fp, 'a').write('\n'.join(lines))


def build_q():
    print('building query...')
    lines = []
    for uid in tqdm(dataset_parser.uids):
        q = dataset_parser.uid2instc[uid]['q']
        if isinstance(q, list):
            q = ' '.join(q)
        lines.append(q)

    q_fp = dataset_root / f'{dataset_var}.query'
    assert not exists(q_fp), f'remove query file first: {q_fp}'
    io.open(q_fp, 'a').write('\n'.join(lines))


def tgt_stats():
    tgt_fp = dataset_root / f'{dataset_var}.tgt'
    nw = []
    lines = io.open(tgt_fp).readlines()

    for line in lines:
        nw.append(len(line.rstrip('\n').split()))

    print(f'min: {min(nw)}, max: {max(nw)}, nw/instance: {sum(nw)/len(lines)}')
    

def overlap_stats(stem):
    """
        Measure the overlaps between src and q.

    """
    def _count_tokens(line, stem):
        tokens = line.strip().split()
        # tokens = [tk for tk in line.strip().split() if tk and tk not in string.punctuation]
        if stem:
            tokens = [porter_stemmer.stem(tk) for tk in tokens]
        count = Counter(tokens)
        return count

    porter_stemmer = PorterStemmer()

    if test_year.startswith('duc'):
        # root_dp = path_parser.duc / f'duc_{dataset_var}.ranked.lines'
        # src_fp = root_dp / f'duc_{dataset_var}.source'
        # q_fp = root_dp / f'duc_{dataset_var}.query'
        src_lines, q_lines = [], []
        for values in dataset_parser.uid2instc.values():
            src_lines.append('\n'.join([' '.join(doc) for doc in values['d']]))
            q_lines.append('\n'.join(values['q']))

    elif test_year.startswith('tdqfs'):
        # root_dp = path_parser.tdqfs / f'tdqfs.ranked.lines'
        # src_fp = root_dp / f'tdqfs.source'
        # q_fp = root_dp / f'tdqfs.query'
        src_lines, q_lines = [], []
        for values in dataset_parser.uid2instc.values():
            src_lines.append('\n'.join([' '.join(doc) for doc in values['d']]))
            q_lines.append('\n'.join(values['q']))
    else:
        src_fp = dataset_root / f'{dataset_var}.source'
        q_fp = dataset_root / f'{dataset_var}.query'
        src_lines = io.open(src_fp).readlines()
        q_lines = io.open(q_fp).readlines()

    # collect stats
    print('collecting stats...')
    overlaps = []
    for src_line, q_line in zip(src_lines, q_lines):
        q_count = _count_tokens(q_line, stem=stem)
        src_count = _count_tokens(src_line, stem=stem)
        overlap = 0
        for k in q_count:
            if k in src_count:
                overlap += 1
        overlaps.append(overlap)
    
    # calculate stats
    # print(overlaps)
    stats = Counter(overlaps)
    # sorted_stats = sorted(stats.items(), key=lambda item: item[0], reverse=False)
    # print(sorted_stats)

    # for k, v in sorted_stats:
    #     ratio = 100 * float(v) / len(overlaps)
    #     print(f'{k}\t{v}\t{ratio:.2f}')
    zero_ratio = 100 * float(stats[0]) / len(overlaps)
    print(f'{test_year}: {stats[0]}/{len(overlaps)}, {zero_ratio:.2f}')
    


def mds_basic_stats():
    ns = 0.0
    nw = 0.0
    nwq = 0.0
    for _, instc in tqdm(dataset_parser.uid2instc.items()):
        sentences = instc['d']
        query = instc['q']
        ns += len(sentences)
        nw += sum([len(word_tokenize(sent)) for sent in sentences])
        nwq += len(word_tokenize(query))
    
    print(f'#sentences/doc: {ns/len(dataset_parser.uid2instc)}')
    print(f'#words/doc: {nw/len(dataset_parser.uid2instc)}')
    print(f'#words/query: {nwq/len(dataset_parser.uid2instc)}')


def mds_paras():
    print('n_clusters:', dataset_parser.n_clusters)
    print('n_paras:', dataset_parser.n_paras)


if __name__ == '__main__':
    # test DebatepediaParser
    # dp = DebatepediaParser()
    # line = dp.proc_line(line='<s> government/privacy : would ids appropriately involve government ? <eos>\n')
    # print(f'line: {line}')

    # build_src()
    # build_tgt()
    # build_q()
    # tgt_stats()

    overlap_stats(stem=True)
    
    # mds_basic_stats()
    # mds_paras()
