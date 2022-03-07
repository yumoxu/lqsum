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

from tqdm import tqdm
import re


class DUCParser:
    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.uid2instc = {}
        self.years = ['2005', '2006', '2007']
        self.sep = '_'
        
        BASE_PAT = '(?<=<{0}> )[\s\S]*?(?= </{0}>)'
        BASE_PAT_WITH_NEW_LINE = '(?<=<{0}>\n)[\s\S]*?(?=\n</{0}>)'

        # query pat
        self.id_pat = re.compile(BASE_PAT.format('num'))
        self.title_pat = re.compile(BASE_PAT.format('title'))
        self.narr_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('narr'))

        # article pat
        self.text_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('TEXT'))
        self.graphic_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('GRAPHIC'))
        self.para_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('P'))

        # preproc configs
        self.doc_proc_config = {'rm_stop': True, 'stem': True}
        self.query_proc_config = {'rm_stop': False, 'stem': True}
        self.summary_proc_config = {'rm_stop': True, 'stem': True}

    def load(self, year):
        cluster_dp = path_parser.duc_cluster / year
        cc_names = [fn for fn in listdir(cluster_dp) if isdir(join(cluster_dp, fn))]
        cc_ids = [self.sep.join((year, cn)) for cn in cc_names]
        
        cid2docs = self.load_doc(year, cc_names)
        cid2query = self.load_query(year)
        cid2summaries = self.load_summary(year, cc_ids)
        
        cc_objs = []
        for cc_name in cc_names:
            cid = self.sep.join((year, cc_name))
            cc_obj = {
                'cid': cid,
                'docs': cid2docs[cid],
                'query': cid2query[cid],
                'summaries': cid2summaries[cid],
            }
            cc_objs.append(cc_obj)

        # following code is for unit testing
        for cid, query in cid2query.items():
            print(f'---cid: {cid}---')
            for k, v in query.items():
                print(f'\t{k}: {v}')

            summaries = cc_obj['summaries']
            doc_objs = cid2docs[cid]

            print(f'\t#docs: {len(doc_objs)}')
            print(f'***first document***')
            for k, v in doc_objs[0].items():
                print(f'\t{k}: {v}') 
            print(f'***first summary***')
            for k, v in summaries[0].items():
                print(f'\t{k}: {v}') 
            
            print(f'***last document***')
            for k, v in doc_objs[-1].items():
                print(f'\t{k}: {v}')
            print(f'***last summary***')
            for k, v in summaries[-1].items():
                print(f'\t{k}: {v}') 
            
        return cc_objs
        
    def load_doc(self, year, cc_names):
        cid2doc = {}
        cluster_dp = path_parser.duc_cluster / year
        
        for cc_name in cc_names:
            cid = self.sep.join((year, cc_name))
            cc_dp = join(cluster_dp, cc_name)

            doc_objs = []
            doc_names = [fn for fn in listdir(cc_dp) if isfile(join(cc_dp, fn))]
            
            for doc_name in doc_names:
                did = self.sep.join((cid, doc_name))
                doc_fp = join(cc_dp, doc_name)
                original_doc, proc_doc = self.doc2sents(doc_fp, **self.doc_proc_config)
                doc_obj  = {
                    'did': did,
                    'original_doc': original_doc,
                    'proc_doc': proc_doc,
                }
                doc_objs.append(doc_obj)
            cid2doc[cid] = doc_objs
        
        return cid2doc

    def load_query(self, year):
        """
            return: cid2query, dict. 
                    keys: "title", "narr", "original_concat", "proc_concat"
        """
        cid2query = {}
        fp = join(path_parser.duc_query, f'{year}.sgml')
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()
        
        segs = article.split('\n\n\n')
        for seg in segs:
            seg = seg.rstrip('\n')
            if not seg:
                continue
            
            query_id = re.search(self.id_pat, seg)
            title = re.search(self.title_pat, seg)
            narr = re.search(self.narr_pat, seg)

            assert query_id, f'no title in {seg}.'
            assert title, f'no title in {seg}.'
            assert narr, f'no narr in {seg}.'

            query_id = query_id.group()
            title = title.group()
            narr = narr.group()

            title = self.remove_extra_spaces(title)
            narr = self.remove_extra_spaces(narr)

            proc_title = self._proc_sent(sent=title, **self.query_proc_config)
            assert proc_title, f'no proc_title in {seg}.'
            
            proc_narr = self._proc_sent(sent=narr, **self.query_proc_config)
            assert proc_narr, f'no proc_narr in {seg}.'

            def _concat(title, narr, is_proc):
                sep = '. '
                if title.endswith('.'):
                    sep = sep[-1]
                
                describe_token = 'Describe '
                if is_proc:
                    describe_token = describe_token.lower()
                title = describe_token + title
                return sep.join((title, narr))
            
            original_concat = _concat(title, narr, is_proc=False)
            proc_concat = _concat(proc_title, proc_narr, is_proc=True)
            
            cid = self.sep.join((year, query_id))
            cid2query[cid] = {
                'title': title, 
                'narr': narr,
                'original_concat': original_concat,
                'proc_concat': proc_concat,
            }
        
        return cid2query

    def load_summary(self, year, cids):
        """
            Load summaries into cid2summary.

            return: cid2summary, dict.
                    Keys: "original", "proc"
        """
        summary_dp = path_parser.duc_summary / year

        cid2summaries = {}
        for cid in cids:
            summaries = []
            for fn in listdir(summary_dp):
                if not fn.startswith(cid):
                    continue
                summary = io.open(join(summary_dp, fn), encoding='latin1').read()
                
                sentences = summary.split('\n')
                sentences = [ss.strip('\n') for ss in sentences if ss.strip('\n')]
                # print(sentences)
                proc_sentences = [self._proc_sent(sent=sent, **self.summary_proc_config) for sent in sentences]
                proc_summary = ' '.join(proc_sentences)
                summary = {
                    'original': summary,
                    'proc': proc_summary,
                }
                summaries.append(summary)
            cid2summaries[cid] = summaries
        
        return cid2summaries

    def _parse_doc_into_paras(self, fp):
        """
            get an article from file.

            return paragraphs.
        """
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()

        pats = [self.text_pat, self.graphic_pat]

        PARA_SEP = '\n\n'

        for pat in pats:
            text = re.search(pat, article)

            if not text:
                continue

            text = text.group()

            # if there is '<p>' in text, gather them to text
            paras = re.findall(self.para_pat, text)
            if paras:
                text = PARA_SEP.join(paras)

            # for text tiling: if paragraph break is a single '\n', double it
            pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
            matches = pattern.finditer(text)
            if not matches:
                text.replace('\n', PARA_SEP)

            if paras:
                return paras
            else:
                return text.split(PARA_SEP)

        logger.warning('No article content in {0}'.format(fp))
        return None

    def _proc_para(self, pp, rm_stop, stem, to_str=False):
        """
            Return both original paragraph and processed paragraph.

        :param pp:
        :param rm_stop:
        :param stem:
        :param to_str: if True, concatenate sentences and return.
        :return:
        """
        original_para_sents, proc_para_sents = [], []

        for ss in sent_tokenize(pp):
            # ss_origin = self._proc_sent(ss, rm_stop=False, stem=False)
            ss_origin = self.remove_extra_spaces(ss)
            ss_proc = self._proc_sent(ss, rm_stop=rm_stop, stem=stem)

            if ss_proc:
                original_para_sents.append(ss_origin)
                proc_para_sents.append(ss_proc)

        if not to_str:
            return original_para_sents, proc_para_sents

        para_origin = ' '.join(original_para_sents)
        para_proc = ' '.join(proc_para_sents)
        return para_origin, para_proc
    
    def doc2sents(self, fp, rm_stop, stem):
        """
        :param fp:
        :param para_org: bool

        :return:
            if para_org=True, 2-layer nested lists;
            else: flat lists.

        """
        
        paras = self._parse_doc_into_paras(fp)
        original_sents, proc_sents = [], []

        if not paras:
            return [], []

        for pp in paras:
            original_para_sents, proc_para_sents = self._proc_para(pp, rm_stop=rm_stop, stem=stem)
            original_sents.extend(original_para_sents)
            proc_sents.extend(proc_para_sents)

        return original_sents, proc_sents

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

    def cid2sents(self, cid, rm_stop=True, stem=True, max_ns_doc=None):
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


class TDQFSParser(DUCParser):
    def __init__(self):
        DUCParser.__init__(self)

    def load(self):
        cid2query, cc_ids = self.load_query()

        cid2docs = self.load_doc(cc_ids)
        cid2summaries = self.load_summary(cc_ids)
        
        cc_objs = []
        for cid in cc_ids:
            cc_obj = {
                'cid': cid,
                'docs': cid2docs[cid],
                'query': cid2query[cid],
                'summaries': cid2summaries[cid],
            }
            cc_objs.append(cc_obj)

        # following code is for unit testing
        for cid, query in cid2query.items():
            print(f'---cid: {cid}---')
            for k, v in query.items():
                print(f'\t{k}: {v}')

            summaries = cc_obj['summaries']
            doc_objs = cid2docs[cid]

            print(f'\t#docs: {len(doc_objs)}')
            print(f'***first document***')
            for k, v in doc_objs[0].items():
                print(f'\t{k}: {v}') 
            print(f'***first summary***')
            for k, v in summaries[0].items():
                print(f'\t{k}: {v}') 
            
            print(f'***last document***')
            for k, v in doc_objs[-1].items():
                print(f'\t{k}: {v}')
            print(f'***last summary***')
            for k, v in summaries[-1].items():
                print(f'\t{k}: {v}') 
            
        return cc_objs

    def load_doc(self, cids):
        cid2doc = {}
        cluster_dp = path_parser.tdqfs_doc
        
        for cid in cids:
            cc_dp = join(cluster_dp, cid)

            doc_objs = []
            doc_names = [fn for fn in listdir(cc_dp) if isfile(join(cc_dp, fn))]
            
            for doc_name in doc_names: 
                did = self.sep.join((cid, doc_name))
                doc_fp = join(cc_dp, doc_name)

                original_sentences = []
                proc_sentences = []
                for line in io.open(doc_fp).readlines():
                    sent = line.strip('\n')
                    if not sent:
                        continue
                    if not sent.endswith('.'):
                        sent += '.'
                    proc_sent = self._proc_sent(sent=sent, **self.doc_proc_config)
                    
                    if proc_sent:
                        original_sentences.append(sent)
                        proc_sentences.append(proc_sent)

                doc_obj  = {
                    'did': did,
                    'original_doc': original_sentences,
                    'proc_doc': proc_sentences,
                }
                doc_objs.append(doc_obj)
            cid2doc[cid] = doc_objs
        
        return cid2doc

    def load_query(self):
        cid2query = {}
        cc_ids = []

        with io.open(path_parser.tdqfs_query, encoding='utf-8') as query_f:
            for line in query_f:
                cid, _, title = line.strip('\n').split('\t')
                cc_ids.append(cid)
                proc_title = self._proc_sent(sent=title, **self.query_proc_config)
                assert proc_title, f'no proc_title in for {cid}'

                cid2query[cid] = {
                    'title': title, 
                    'proc_title': proc_title,
                }
        
        return cid2query, cc_ids

    def load_summary(self, cids):
        """
            Load summaries into cid2summary.

            return: cid2summary, dict.
                    Keys: "original", "proc"
        """
        summary_dp = path_parser.tdqfs_summary
        cid2summaries = {}
        for cid in cids:
            summaries = []
            _dp = summary_dp / cid
            for fn in listdir(_dp):
                summary = io.open(join(_dp, fn)).read()
                
                sentences = summary.split('\n')
                sentences = [ss.strip('\n') for ss in sentences if ss.strip('\n')]
                proc_sentences = [self._proc_sent(sent=sent, **self.summary_proc_config) for sent in sentences]
                proc_summary = ' '.join(proc_sentences)
                summary = {
                    'original': summary,
                    'proc': proc_summary,
                }
                summaries.append(summary)
            cid2summaries[cid] = summaries
        
        return cid2summaries


def build_duc():
    dp = DUCParser()
    for year in dp.years:
        dump_fp =  path_parser.duc / f'duc_{year}.json'
        cc_objs = dp.load(year=year)
        
        with open(dump_fp, "a") as f:
            json_objs = [json.dumps(cc_obj) for cc_obj in cc_objs]
            f.write('\n'.join(json_objs))
        print(f'Dump duc data to: {dump_fp}')


def build_tdqfs():
    dp = TDQFSParser()
    dump_fp = path_parser.tdqfs / f'tdqfs.json'
    cc_objs = dp.load()
    
    with open(dump_fp, "a") as f:
        json_objs = [json.dumps(cc_obj) for cc_obj in cc_objs]
        f.write('\n'.join(json_objs))
    print(f'Dump tdqfs data to: {dump_fp}')


if __name__ == '__main__':
    # build_duc()
    build_tdqfs()
