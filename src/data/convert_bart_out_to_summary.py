import io
import json
import sys
import os
from os.path import dirname, abspath, exists, split

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from utils.config_loader import path_parser
from data.mds_parser import DUCParser
import summ.compute_rouge as rouge

import utils.tools as tools
from nltk.tokenize import sent_tokenize, word_tokenize


class Converter:
    def __init__(self, year, model_id, ckpt, guid_name, tag_mode, min_max, postproc_mode='', bart_out_name=None):
        self.year = year
        self.postproc_mode = postproc_mode
        if bart_out_name:
            self.bart_out_name = bart_out_name
        else:
            self.bart_out_name = f'model_{model_id}_{ckpt}-bpeTags-duc_{year}-source-{guid_name}.{tag_mode}.{min_max}'
    
    def load_bart_out(self):
        bart_fp = path_parser.bart_out / self.bart_out_name
        bart_lines = io.open(bart_fp).readlines()
        bart_lines = [line.strip('\n') for line in bart_lines]
        return bart_lines

    def load_meta_info(self):
        meta_fp = path_parser.duc / f'duc_{self.year}.ranked.lines' / f'duc_{self.year}.meta'
        meta_info = []
        with io.open(meta_fp) as meta_f:
            for line in meta_f:
                cid, n_docs = line.strip('\n').split('\t')
                n_docs = int(n_docs)
                meta_info.append((cid, n_docs))

        return meta_info

    def build_clusters(self, bart_lines, meta_info):
        """
            cluster: cid to summary sentences
        """
        clusters = {}
        start = 0
        for cid, n_docs in meta_info:
            end = start + n_docs
            clusters[cid] = bart_lines[start:end]
            start = end
        
        return clusters

    def post_proc(self, clusters):
        cid2sentences = {}
        
        for cid, docs in clusters.items():
            sentences = []
            for doc in docs:
                sentences.extend(sent_tokenize(doc))
            cid2sentences[cid] = sentences

            print(f'cid: {cid}, senteces: {len(sentences)}')
        
        assert self.postproc_mode == ''
        return cid2sentences

    def log(self, cid2sentences):
        for cid, sentences in cid2sentences.items():
            print(f'*** cid: {cid} ***')
            print('\n'.join(sentences))
    
    def compose(self):
        bart_lines = self.load_bart_out()
        meta_info = self.load_meta_info()
        
        clusters = self.build_clusters(bart_lines, meta_info)
        cid2sentences = self.post_proc(clusters)

        # self.log(cid2sentences)

        return cid2sentences


class MargeConverter:
    def __init__(self, year, model_id, ckpt, guid_name, tag_mode, min_max, bart_out_name=None):
        self.year = year
        if bart_out_name:
            self.bart_out_name = bart_out_name
        else:
            self.bart_out_name = f'model_{model_id}_{ckpt}-bpeTags-duc_{year}-source-{guid_name}.{tag_mode}.{min_max}'
    
    def load_bart_out(self):
        bart_fp = path_parser.bart_out / self.bart_out_name
        bart_lines = io.open(bart_fp).readlines()
        bart_lines = [line.strip('\n') for line in bart_lines]
        return bart_lines

    def load_meta_info(self):
        meta_fp = path_parser.duc / f'duc_{self.year}.marge.lines' / f'duc_{self.year}.meta'
        meta_info = []
        with io.open(meta_fp) as meta_f:
            for line in meta_f:
                cid = line.strip('\n')
                meta_info.append((cid))

        return meta_info
    
    def compose(self):
        bart_lines = self.load_bart_out()
        meta_info = self.load_meta_info()
        return list(zip(meta_info, bart_lines))


class TDQFSConverter(Converter):
    def __init__(self, model_id, ckpt, guid_name, tag_mode, min_max, postproc_mode='', bart_out_name=None):
        self.postproc_mode = postproc_mode
        if bart_out_name:
            self.bart_out_name = bart_out_name
        else:
            self.bart_out_name = f'model_{model_id}_{ckpt}-bpeTags-tdqfs-source-{guid_name}.{tag_mode}.{min_max}'

    def load_meta_info(self):
        meta_fp = path_parser.tdqfs / 'tdqfs.ranked.lines' / 'tdqfs.meta'
        meta_info = []
        with io.open(meta_fp) as meta_f:
            for line in meta_f:
                cid, n_docs = line.strip('\n').split('\t')
                n_docs = int(n_docs)
                meta_info.append((cid, n_docs))

        return meta_info


class Selector:
    def __init__(self, cid, sentences, text_dp, cos_threshold, max_n_summary_words):
        self.cid = cid
        self.cos_threshold = cos_threshold
        self.duc_parser = DUCParser()

        self.text_fp = text_dp / cid  # for dumping summaries

        self.sid2sent = {}
        self.ordered_sids = []
        for i, sentence in enumerate(sentences):
            self.ordered_sids.append(i)
            self.sid2sent[i] = sentence

        self.max_n_summary_words = max_n_summary_words
        print(f'max_nw for {cid}: {self.max_n_summary_words}')

        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            return True

        if self.cos_threshold == 1.0:
            return True

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _select_sent_ids(self):
        selected_sids = []
        n_total_words = 0

        for sid in self.ordered_sids:
            cand_sent_original = self.sid2sent[sid]
            cand_sent_proc = self.duc_parser._proc_sent(cand_sent_original, rm_stop=True, stem=True)
            cand_sent_words = word_tokenize(cand_sent_proc)

            if not self._sim_cond(cand_sent_words):
                continue

            self.summary_sent_words.append(cand_sent_words)

            selected_sids.append(sid)
            n_total_words += len(word_tokenize(cand_sent_original))  # add the genuine #words in original sent

            num_words_to_clip = n_total_words - self.max_n_summary_words
            if num_words_to_clip >= 0:
                break

        return selected_sids

    def gen_and_dump_summary(self):
        sids = self._select_sent_ids()
        wc = 0
        break_when_finish_flag = False
        selected_sents = []
        for sid in sids:
            cand_sent = self.sid2sent[sid]
            selected_sents.append(cand_sent)

            cand_words = word_tokenize(cand_sent)
            wc += len(cand_words)

            if break_when_finish_flag:
                break
            if wc >= self.max_n_summary_words:  # break in next iter to get one more additional sentence
                break_when_finish_flag = True

        assert selected_sents, f'selected_sents: {selected_sents}'
        summary = '\n'.join(selected_sents)
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)


def _rank_files(text_dp, ref_dp):
    text_fns = [fn for fn in os.listdir(text_dp)]
    fn2rouge2 = {}
    for fn in text_fns:
        rouge_out = rouge.compute_rouge_lqsum_per_file(text_dp=text_dp, ref_dp=ref_dp, text_fn=fn,
            split_sentences=True, verbose_output=False)
        fn2rouge2[fn] = float(rouge_out.split('\t')[1])
    
    ranked_items = sorted(fn2rouge2.items(), key=lambda x:x[1], reverse=True)
    ranked_files, ranked_scores = zip(*ranked_items)
    file_str = '\n'.join(ranked_files)
    print(file_str)
    print('**********ranked scores**********') 
    score_str = '\n'.join([str(score) for score in ranked_scores])
    print(score_str)
    
    temp_dp = path_parser.proj_root / 'temp'
    rank_fp = temp_dp/f'{text_dp}.rank'
    score_fp = temp_dp/f'{text_dp}.score'
    io.open(rank_fp).write(file_str)
    io.open(score_fp).write(score_str)
    print(f'Dump rank records to: {rank_fp}')


def convert_to_summary(year, model_id, ckpt, guid_name, tag_mode, min_max, postproc_mode, 
        bart_out_name=None,
        cos_threshold=0.6, max_n_summary_words=500, rouge_only=False, rank_rouge=True):

    if year == 'tdqfs':
        converter = TDQFSConverter(model_id, ckpt, guid_name, tag_mode, min_max, 
            postproc_mode=postproc_mode, bart_out_name=bart_out_name)
        ref_dp = path_parser.tdqfs_summary_targets
    else:
        converter = Converter(year, model_id, ckpt, guid_name, tag_mode, min_max, 
            postproc_mode=postproc_mode, bart_out_name=bart_out_name)
        ref_dp = path_parser.duc_summary / year
        
    text_dn = f'{converter.bart_out_name}.{cos_threshold}_cos.{max_n_summary_words}_words'
    if postproc_mode:
        text_dn += f'.post_{postproc_mode}'
    text_dp = path_parser.bart_out / text_dn
    
    if not rouge_only:
        cid2sentences = converter.compose()

        if exists(text_dp):
            raise  ValueError(f'Remove text_dp to continue: {text_dp}')
        os.mkdir(text_dp)

        for cid, sentences in cid2sentences.items():
            selector = Selector(cid, sentences, text_dp, cos_threshold, max_n_summary_words)
            selector.gen_and_dump_summary()

    if rank_rouge:
        _rank_files(text_dp, ref_dp)
        return 

    rouge_out = rouge.compute_rouge_lqsum(text_dp=text_dp, ref_dp=ref_dp, 
        split_sentences=False, verbose_output=False)
    print(rouge_out)


def convert_marge_to_summary(year, model_id, ckpt, guid_name, tag_mode, min_max,
        cos_threshold=0.6, max_n_summary_words=500, rouge_only=False):
    
    converter = MargeConverter(year, model_id, ckpt, guid_name, tag_mode, min_max)

    text_dn = f'{converter.bart_out_name}.{cos_threshold}_cos.{max_n_summary_words}_words'
    text_dp = path_parser.bart_out / text_dn
    
    if not rouge_only:
        cid_summary_pairs = converter.compose()

        if exists(text_dp):
            raise  ValueError(f'Remove text_dp to continue: {text_dp}')
        os.mkdir(text_dp)

        for cid, summary in cid_summary_pairs:
            text_fp = text_dp / cid
            with open(text_fp, mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)
    
    ref_dp = path_parser.duc_summary / year
    # rouge_out = rouge.compute_rouge_lqsum(text_dp=text_dp, ref_dp=ref_dp, 
    #     split_sentences=True, verbose_output=True)
    _rank_files(text_dp, ref_dp)


if __name__ == '__main__':
    # bart_out_name:
    # LaqSum: model_64_10-bpeTags-duc_2006-source-with_bpe_lcs_tags.query_11.min10max60
    # LaqSum: model_64_10-bpeTags-duc_2007-source-with_bpe_lcs_tags.query_11.min10max60
    # LaqSum: model_64_10-bpeTags-tdqfs-source-with_bpe_lcs_tags_and_qe_1.query_11.min10max60
    # GSum: model_17-grsum-ducdoc_2006-0.6_cos-0_wan-ns_3.min10max60
    # GSum: model_17-grsum-ducdoc_2007-0.6_cos-0_wan-ns_3.min10max60
    # GSum: model_17-grsum-tdqfsdoc-0.6_cos-0_wan-ns_3.min10max60
    params = {
        'year': '2007',  # if bart_out_name is set, this has to be set to be consistent: 2006, 2007, or tdqfs
        'model_id': 64,
        'ckpt': 10,
        'guid_name': 'with_bpe_lcs_tags',
        'tag_mode': 'query_11',  # query_0.99, query_11
        'min_max': 'min10max60', # min55max140, min10max60
        'postproc_mode': '',
        'bart_out_name': 'model_17-grsum-ducdoc_2007-0.6_cos-0_wan-ns_3.min10max60',
        'cos_threshold': 0.6,
    }
    convert_to_summary(**params, rouge_only=True, rank_rouge=True)

    # params = {
    #     'year': 'tdqfs',
    #     'model_id': 64,
    #     'ckpt': 10,
    #     'guid_name': 'with_bpe_lcs_tags',  # with_bpe_lcs_tags_marge (?)
    #     'tag_mode': 'query_11',  # query_0.99, query_11
    #     'min_max': 'min10max60', # min55max140, min10max60, min300max400
    #     'cos_threshold': 0.6,
    # }
    # convert_marge_to_summary(**params, rouge_only=True)
