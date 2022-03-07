# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

import io
from tqdm import tqdm
from data.dataset_parser import dataset_parser
import utils.config_loader as config
from utils.config_loader import logger
import utils.tools as tools
import summ.compute_rouge as rouge
import nltk
from ir.ir_tools import load_retrieved_sentences


class SelectorforELI5:
    def __init__(self,
                 cid,
                 rank_fp,
                 text_dp,
                 cos_threshold,
                 rm_dialog,
                 max_n_summary_sentences,
                 retrieved_dp=None):
        self.cid = cid
        self.cos_threshold = cos_threshold
        self.word_tokenize = nltk.tokenize.word_tokenize

        # fps for rank and text
        self.rank_fp = rank_fp
        if not exists(self.rank_fp):
            raise ValueError('rank_fp does not exist: {}'.format(self.rank_fp))

        self.text_fp = join(text_dp, cid)  # for dumping summaries

        if retrieved_dp:
            self.original_sents, self.processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
        else:
            self.original_sents, self.processed_sents = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog)

        if max_n_summary_sentences:
            self.max_n_summary_sentences = max_n_summary_sentences

        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _load_ranking(self):
        with io.open(self.rank_fp, encoding='utf-8') as f:
            content = f.readlines()

        ranked_sent_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]
        return ranked_sent_ids

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            return True

        if self.cos_threshold == 1.0:
            return True

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _get_sent(self, sents, sid):
        if '_' in sid:
            return tools.get_sent(sents, sid)
        else:
            return sents[int(sid)]

    def _select_sent_ids(self):
        ranked_sent_ids = self._load_ranking()
        budget = len(ranked_sent_ids) - self.max_n_summary_sentences
        if budget <= 0:
            return ranked_sent_ids
        
        selected_sids = []
        for idx, sid in enumerate(ranked_sent_ids):
            if budget <= 0:
                selected_sids.extend(ranked_sent_ids[idx:])
                return selected_sids
            
            cand_sent_original = self._get_sent(self.original_sents, sid)
            cand_sent_proc = self._get_sent(self.processed_sents, sid)
            cand_sent_words = self.word_tokenize(cand_sent_proc)
            
            if not self._sim_cond(cand_sent_words):
                budget -= 1
                continue

            self.summary_sent_words.append(cand_sent_words)
            selected_sids.append(sid)
        
        return selected_sids

    def _gen_summary_wo_tokenize(self):
        sids = self._select_sent_ids()
        # logger.info('sids: {}'.format(sids))
        # wc = 0
        selected_sents = []
        for sid in sids:
            cand_sent = self._get_sent(self.original_sents, sid).strip(' ')
            selected_sents.append(cand_sent)
            # cand_words = self.word_tokenize(cand_sent)
            # wc += len(cand_words)
            if len(selected_sents) == self.max_n_summary_sentences:
                break

        summary = '\n'.join(selected_sents)
        return summary

    def gen_and_dump_summary(self):
        summary = self._gen_summary_wo_tokenize()
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)


def select_end2end_for_eli5(model_name,
        n_iter=None,
        length_budget_tuple=None,
        diversity_param_tuple=None,
        cos_threshold=None,
        extra=None,
        rank_model_name=None,
        rel_sents_dp=None,
        retrieved_dp=None,
        rm_dialog=True,
        cc_ids=None
        ):
    """

    :param model_name:
    :param n_iter:
    :param cos_threshold: 0.5, 0.6
    :param max_n_summary_words: 500
    :param rank_model_name: you can specify rank_model_name; default is set to model_name.
    :param rm_dialog: only useful when retrieved_dp=None
    :return:
    """
    # make dump dir
    text_params = {
        'model_name': model_name,
        'cos_threshold': cos_threshold,
        'n_iter': n_iter,
        'diversity_param_tuple': diversity_param_tuple,
        'length_budget_tuple': length_budget_tuple,
        'extra': extra,
    }

    text_dp = tools.init_text_dp_for_eli5(**text_params)

    base_selector_params = {
        'text_dp': text_dp,
        'cos_threshold': cos_threshold,
        'rel_sents_dp': rel_sents_dp,
        'retrieved_dp': retrieved_dp,
    }

    budget_type, budget = length_budget_tuple
    if budget_type == 'ns':
        SelectorCls = SelectorforELI5
        base_selector_params['max_n_summary_sentences'] = budget 
    elif budget_type == 'nw':
        SelectorCls = Selector  
        base_selector_params['max_n_summary_words'] = 500  # make sure the budget is satisified but only first 250 will be evaluated
        base_selector_params['ngram_block'] = False
    else:
        raise ValueError(f'Invalid budget_type: {budget_type}')

    # logger.info('[SELECT SENTS] selecting sents for {} clusters'.format(len(cc_ids)))
    if not rank_model_name:
        rank_model_name = model_name

    rank_dp = tools.get_rank_dp(rank_model_name,
                                n_iter=n_iter,
                                diversity_param_tuple=diversity_param_tuple,
                                extra=extra)

    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
            'rm_dialog': rm_dialog,
        }

        selector = SelectorCls(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))
