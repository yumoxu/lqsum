import io
import re
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from tqdm import tqdm
from os.path import isfile, isdir, join, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from data.dataset_parser import DatasetParser
import utils.config_loader as config
from utils.config_loader import logger, path_parser
import utils.tools as tools
import itertools


def print_dist(data_list, stat_var, end, higher_var=None, n_bins=10):
    if not higher_var:
        if stat_var == 'sent':
            higher_var = 'doc'
        elif stat_var == 'word':
            higher_var = 'sent'
        else:
            raise ValueError('Invalid stat_var: {}'.format(stat_var))

    data = np.array(data_list)
    total = float(len(data_list))
    logger.info('total #{0}: {1}'.format(higher_var, total))
    hist, bins = np.histogram(data, bins=np.linspace(0, end, n_bins, endpoint=False, dtype=np.int32))
    ratios = hist / total

    intervals = ['{0}-{1}'.format(bins[i], bins[i + 1]) for i in range(len(hist))]

    for i, h, r in zip(intervals, hist, ratios):
        logger.info('#{0} whose #{1} belongs to {2}: {3}, {4:2f}%'.format(higher_var, stat_var, i, h, r * 100))

    left = (1 - sum(ratios)) * 100
    logger.info(
        '#{0} whose #{1} belongs to >{2}: {3}, {4:2f}%'.format(higher_var, stat_var, bins[-1], total - sum(hist),
                                                               left))

    ratios = [str(r * 100) for r in ratios.tolist()]
    ratios.append(str(left))
    logger.info('\t'.join(ratios))


class StatsPrinter:
    def __init__(self):
        self.dataset_parser = DatasetParser()
        self.dp_data_docs = path_parser.data_docs
        self.N_THREADS = 6

    def print_cc_stats(self):
        years = [2005, 2006]
        for year in years:
            dp_duc = join(self.dp_data_docs, 'duc{}_docs'.format(year))
            clusters = [cluster for cluster in listdir(dp_duc) if isdir(join(dp_duc, cluster))]
            n_clusters = len(clusters)
            n_docs = 0

            for cluster in clusters:
                dp_cluster = join(dp_duc, cluster)
                fns = [fn for fn in listdir(dp_cluster) if isfile(join(dp_cluster, fn))]
                n_docs += len(fns)

            avg_docs = float(n_docs) / n_clusters

            stat_pattern = '{0}: {1}: #clusters: {2}, #docs: {3}, #avg_docs_per_clusters: {4:2f}'
            logger.info(stat_pattern.format('duc', year, n_clusters, n_docs, avg_docs))

    def print_avg_n_stats(self):
        n_sents_list, n_words_in_doc_list, n_words_in_sent_list, max_n_words_in_sent = list(), list(), list(), 0

        n_docs = 0
        for year in config.years:
            doc_fps = tools.get_doc_fps_yearly(year)
            n_docs += len(doc_fps)

            for doc_fp in tqdm(doc_fps):
                sents = self.dataset_parser.doc2sents(doc_fp, para_org=False)
                n_sents_list.append(len(sents))
                n_words_this = [len(sent) for sent in sents]

                n_words_in_doc_list.append(sum(n_words_this))

                new_max = max(n_words_this)
                if new_max > max_n_words_in_sent:
                    max_n_words_in_sent = new_max

        total_n_sents = float(sum(n_sents_list))
        max_n_sents = float(max(n_sents_list))
        total_n_words = float(sum(n_words_in_doc_list))

        avg_n_sents_in_doc = total_n_sents / n_docs
        avg_n_words_in_doc = total_n_words / n_docs
        avg_n_words_in_sent = total_n_words / total_n_sents

        logger.info('#docs: {}'.format(n_docs))
        logger.info('#sents in doc: AVG - {0:.2f}, MAX - {1:.2f}'.format(avg_n_sents_in_doc, max_n_sents))
        logger.info('#words in sent: AVG - {0:.2f}, MAX - {1:.2f}'.format(avg_n_words_in_sent, max_n_words_in_sent))
        logger.info('#words in doc: AVG - {0:.2f}'.format(avg_n_words_in_doc))

    @staticmethod


    def print_word_sent_dist_in_doc(self):
        n_sents_list, n_words_list = list(), list()
        for year in config.years:
            doc_fps = tools.get_doc_fps_yearly(year)
            for doc_fp in tqdm(doc_fps):
                sents = self.dataset_parser.doc2sents(doc_fp, para_org=False)
                n_sents_list.append(len(sents))

                n_words = [len(self.dataset_parser.sent2words(sent)) for sent in sents]
                n_words_list.extend(n_words)

        max_n_sents, max_n_words = 400, 100
        self.print_dist(n_sents_list, stat_var='sent', end=max_n_sents)
        self.print_dist(n_words_list, stat_var='word', end=max_n_words)

    def print_sent_dist_in_trigger(self):
        triggers = self.dataset_parser.get_trigger_list(tokenize_narr=True)
        n_sents_list = [len(tt) for tt in triggers]
        # for tt in triggers:
        #     print(tt)

        max_n_sents = 10
        self.print_dist(n_sents_list, stat_var='sent', end=max_n_sents)

        n_words_list = [len(self.dataset_parser.trigger_sent2words(sent)) for tt in triggers for sent in tt]
        max_n_words = 100
        self.print_dist(n_words_list, stat_var='word', end=max_n_words)

    def print_word_dist_in_trigger(self):
        triggers = self.dataset_parser.get_trigger_list(tokenize_narr=False)
        n_words_list = [len(self.dataset_parser.sent2words(tt)) for tt in triggers]

        max_n_words = 200
        self.print_dist(n_words_list, stat_var='word', end=max_n_words)

    def print_word_sent_dist_in_paragraph(self, year):
        """
            print #paras in docs, and #words and #sents in paragraphs.
        """
        doc_fps = tools.get_doc_fps_yearly(year)
        n_paras_list, n_sents_list, n_words_list = [], [], []
        for doc_fp in doc_fps:
            # paragraphs = self.dataset_parser.parse_article(doc_fp, concat_paras=False, clip=False)
            paras = self.dataset_parser.get_doc(doc_fp, concat_paras=False)
            n_paras_list.append(len(paras))
            for pp in paras:
                n_sents_list.append(len(pp))
                n_words_list.append(sum([len(sent) for sent in pp]))

        max_n_paras, max_n_sents, max_n_words = 30, 30, 500
        self.print_dist(n_paras_list, stat_var='para', higher_var='doc', end=max_n_paras)
        self.print_dist(n_sents_list, stat_var='sent', higher_var='para', end=max_n_sents)
        self.print_dist(n_words_list, stat_var='word', higher_var='para', end=max_n_words)

    def print_word_sent_dist_in_query(self, year):
        n_sents_list, n_words_list = [], []

        # narr
        pos_narr_info = self.dataset_parser.build_query_info(year, tokenize=None)
        neg_narr_info = self.dataset_parser.build_neg_query_dict(year, query_type=config.NARR)

        for is_neg, n_info in enumerate((pos_narr_info, neg_narr_info)):
            if is_neg:
                narrs = n_info.values()
                narrs = list(itertools.chain(*narrs))  # each value is a list. could have two neg (one altered, one sampled).
            else:
                narrs = [v[config.NARR] for v in n_info.values()]

            for narr in narrs:  # each narr is a paragraph
                # logger.info('is_neg: {0}, narr: {1}'.format(is_neg, narr))
                narr = self.dataset_parser.parse_para(narr, clip=False)
                n_sents_list.append(len(narr))
                n_words_list.append(sum([len(nr) for nr in narr]))

        # headline
        pos_headline_info = self.dataset_parser.build_headline_info(year)
        neg_headline_info = self.dataset_parser.build_neg_query_dict(year, query_type=config.HEADLINE)
        for is_neg, h_info in enumerate((pos_headline_info, neg_headline_info)):
            if is_neg:
                headlines = h_info.values()  # each value is a list. could have two neg (one altered, one sampled).
                headlines = list(itertools.chain(*headlines))
            else:
                headlines = [v[config.HEADLINE] for v in h_info.values()]

            for hl in headlines:  # each headline is a sentence
                n_sents_list.append(1)
                n_words_list.append(len(hl))

        max_n_sents, max_n_words = 10, 200
        self.print_dist(n_sents_list, stat_var='sent', higher_var='query', end=max_n_sents)
        self.print_dist(n_words_list, stat_var='word', higher_var='query', end=max_n_words, n_bins=20)

    def plot_f1_by_doms(self, doc_f1, sent_f1, mode='color'):
        fig, ax = plt.subplots()
        index = np.arange(len(self.doms_final))
        bar_width = 0.35

        if mode == 'bw':
            ax.bar(index, doc_f1, bar_width, edgecolor='black', facecolor='w', label='Document', hatch='//')
            ax.bar(index + bar_width, sent_f1, bar_width, edgecolor='black', facecolor='w', label='Sentence',
                   hatch='\\\\')
        else:
            ax.bar(index, doc_f1, bar_width, color='#5B3D1F', label='Document')
            ax.bar(index + bar_width, sent_f1, bar_width, color='#B0763D', label='Sentence')

        ax.set_title('Domain Macro-F1 Scores')
        ax.set_xlabel('Domains', fontsize=12)
        ax.set_ylabel('Macro-F1', fontsize=12)
        ax.set_ylim((0.0, 1.0))

        plt.xticks(index, self.doms_final, fontsize=12, rotation=0)

        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(self.doms_final)
        ax.legend()

        # annotate
        for (pos, d_f1, s_f1) in zip(index, doc_f1, sent_f1):
            ax.annotate('{:.6f}'.format(d_f1), xy=(pos - 0.27, d_f1 + 0.01), fontsize=10)
            ax.annotate('{:.6f}'.format(s_f1), xy=(pos + 0.17, s_f1 + 0.01), fontsize=10)

        fig.tight_layout()
        plt.show()

    def print_human_eval_stats(self):
        en_nyt_csv_fps = [join(path_parser.en_nyt_csv, fn)
                          for fn in listdir(path_parser.en_nyt_csv) if isfile(join(path_parser.en_nyt_csv, fn))]
        en_wiki_csv_fps = [join(path_parser.en_wiki_csv, fn)
                           for fn in listdir(path_parser.en_wiki_csv) if isfile(join(path_parser.en_wiki_csv, fn))]
        zh_wiki_csv_fps = [join(path_parser.zh_wiki_csv, fn)
                           for fn in listdir(path_parser.zh_wiki_csv) if isfile(join(path_parser.zh_wiki_csv, fn))]

        for fps in (en_nyt_csv_fps, en_wiki_csv_fps, zh_wiki_csv_fps):
            total_n_sents = 0
            for fp in fps:
                with io.open(fp, encoding='utf-8') as f:
                    n_sents = f.readlines()[-1].split('\t')[0]
                    if n_sents == '8':
                        print(fp)
                    total_n_sents += int(n_sents)
            logger.info('n_sents: {}'.format(total_n_sents))


if __name__ == '__main__':
    year = '2005'
    stat_printer = StatsPrinter()
    # stat_printer.print_cc_stats()
    # stat_printer.print_avg_n_stats()
    # stat_printer.print_word_sent_dist_in_doc()
    # stat_printer.print_word_dist_in_trigger()
    stat_printer.print_sent_dist_in_trigger()
    # stat_printer.print_word_sent_dist_in_paragraph(year)
    # stat_printer.print_word_sent_dist_in_query(year)
