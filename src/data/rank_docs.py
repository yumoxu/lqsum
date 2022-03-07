import io
import json
import sys
from os.path import dirname, abspath

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from utils.config_loader import path_parser
import utils.tfidf_tools as tfidf_tools
import summ.rank_sent as rank_sent


class DUCDocRanker:
    def __init__(self):
        self.years = ['2005', '2006', '2007']

    def load(self, year):
        dump_fp =  path_parser.duc / f'duc_{year}.json'
        cc_objs = [json.loads(line.strip('\n')) for line in io.open(dump_fp).readlines()]
        return cc_objs

    def _rank(self, query, docs):
        """
            Rank `docs` with `query` using Term Frequency.

            return: a list of ranked items (did, score).
        """
        proc_docs = [' '.join(doc['proc_doc']) for doc in docs]
        proc_query = query['proc_concat']
        rel_scores = tfidf_tools._compute_rel_scores_tf([proc_docs], proc_query)  
        print('rel_scores: {}'.format(rel_scores))
        assert len(rel_scores) == len(proc_docs), f'#rel_scores: {len(rel_scores)}, #proc_docs: {len(proc_docs)}'

        did2score = {}
        dids = [doc['did'] for doc in docs]
        for did, score in zip(dids, rel_scores):
            did2score[did] = score

        # rank scores
        did_score_list = rank_sent.sort_sid2score(did2score)
        return did_score_list

    def log(self, cc_objs):
        for cc_obj in cc_objs:
            cid = cc_obj['cid']
            query = cc_obj['query']

            print(f'---cid: {cid}---')
            for k, v in query.items():
                print(f'\t{k}: {v}')

            # summaries = cc_obj['summaries']
            doc_objs = cc_obj['docs']

            print(f'\t#docs: {len(doc_objs)}')
            print(f'\tdoc rank: {cc_obj["rank"]}')

            print(f'***first document***')
            for k, v in doc_objs[0].items():
                print(f'\t{k}: {v}') 
            # print(f'***first summary***')
            # for k, v in summaries[0].items():
            #     print(f'\t{k}: {v}') 
            
            print(f'***last document***')
            for k, v in doc_objs[-1].items():
                print(f'\t{k}: {v}') 
            # print(f'***last summary***')
            # for k, v in summaries[-1].items():
            #     print(f'\t{k}: {v}') 
        
    def rank(self, year):
        cc_objs = self.load(year)
        for cc_obj in cc_objs:
            doc_objs = cc_obj['docs']
            did_score_list = self._rank(query=cc_obj['query'], docs=doc_objs)
            cc_obj['rank'] = did_score_list
            
            # re-order doc objs
            did2doc_obj = {}
            for doc_obj in doc_objs:
                did2doc_obj[doc_obj['did']] = doc_obj
            
            ranked_doc_objs = [did2doc_obj[did] for did, _ in did_score_list]
            cc_obj['docs'] = ranked_doc_objs

        # following code is for unit testing
        self.log(cc_objs)
        return cc_objs


class TDQFSDocRanker(DUCDocRanker):
    def __init__(self):
        DUCDocRanker.__init__(self)

    def load(self):
        dump_fp =  path_parser.tdqfs / f'tdqfs.json'
        cc_objs = [json.loads(line.strip('\n')) for line in io.open(dump_fp).readlines()]
        return cc_objs

    def _rank(self, query, docs):
        """
            Rank `docs` with `query` using Term Frequency.

            return: a list of ranked items (did, score).
        """
        proc_docs = [' '.join(doc['proc_doc']) for doc in docs]  # proc_doc are sentences
        proc_query = query['proc_title']

        # from collections import Counter
        # words_q = Counter(proc_query.split())
        # for pd in proc_docs:
        #     words_d = Counter(pd.split())
        #     overlap = 0
        #     for wq in words_q:
        #         overlap += words_d.get(wq, 0)
        #     print(f'overlap: {overlap}')

        rel_scores = tfidf_tools._compute_rel_scores_tf([proc_docs], proc_query)  
        print('rel_scores: {}'.format(rel_scores))
        assert len(rel_scores) == len(proc_docs), f'#rel_scores: {len(rel_scores)}, #proc_docs: {len(proc_docs)}'

        did2score = {}
        dids = [doc['did'] for doc in docs]
        for did, score in zip(dids, rel_scores):
            did2score[did] = score

        # rank scores
        did_score_list = rank_sent.sort_sid2score(did2score)
        return did_score_list

    def rank(self):
        cc_objs = self.load()
        for cc_obj in cc_objs:
            doc_objs = cc_obj['docs']
            did_score_list = self._rank(query=cc_obj['query'], docs=doc_objs)
            cc_obj['rank'] = did_score_list
            
            # re-order doc objs
            did2doc_obj = {}
            for doc_obj in doc_objs:
                did2doc_obj[doc_obj['did']] = doc_obj
            
            ranked_doc_objs = [did2doc_obj[did] for did, _ in did_score_list]
            cc_obj['docs'] = ranked_doc_objs

        self.log(cc_objs)
        return cc_objs


def build_duc():
    dr = DUCDocRanker()
    for year in dr.years:
        dump_fp =  path_parser.duc / f'duc_{year}.ranked'
        cc_objs = dr.rank(year=year)
        
        with open(dump_fp, "a") as f:
            json_objs = [json.dumps(cc_obj) for cc_obj in cc_objs]
            f.write('\n'.join(json_objs))
        print(f'Dump ranked duc data to: {dump_fp}')


def build_tdqfs():
    dr = TDQFSDocRanker()
    dump_fp =  path_parser.tdqfs / f'tdqfs.ranked'
    cc_objs = dr.rank()
    
    with open(dump_fp, "a") as f:
        json_objs = [json.dumps(cc_obj) for cc_obj in cc_objs]
        f.write('\n'.join(json_objs))
    print(f'Dump ranked tdqfs data to: {dump_fp}')
    


if __name__ == '__main__':
    # build_duc()
    build_tdqfs()
