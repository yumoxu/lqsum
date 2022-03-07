import io
import json
import sys
import os
from os.path import dirname, abspath, exists

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from utils.config_loader import path_parser
from pathlib import Path


class DUCFormatter:    
    def __init__(self):
        self.years = ['2005', '2006', '2007']

    def format_as_multiple_files(self, year):
        src_fp = path_parser.duc / f'duc_{year}.ranked'
        dump_dp = path_parser.duc / f'duc_{year}.ranked.lines'

        if exists(dump_dp):
            raise  ValueError(f'Remove the target directory to continue: {dump_dp}')
        os.mkdir(dump_dp)
        
        cc_lines = io.open(src_fp).readlines()
        for line in cc_lines:
            cc_obj = json.loads(line.strip('\n'))
            cid = cc_obj['cid']
            query = cc_obj['query']['original_concat']
            docs = [doc_obj['original_doc'] for doc_obj in cc_obj['docs']]
            doc_fp = dump_dp / f'{cid}.source'
            query_fp = dump_dp / f'{cid}.query'
            with io.open(query_fp, 'a') as query_f, io.open(doc_fp, 'a') as doc_f:
                for doc in docs:
                    doc_f.write(' '.join(doc) + '\n')  # each doc is a list of sentences
                    query_f.write(query + '\n')
            
            lines_q = len(io.open(query_fp).readlines())
            lines_d = len(io.open(doc_fp).readlines())
            assert lines_q == lines_d, f'line_q: {lines_q} != line_d: {lines_d}'

        print(f'Dump bart-formatted mds data to: {dump_dp}')

    def format_as_single_file(self, year):
        src_fp = path_parser.duc / f'duc_{year}.ranked'
        dump_dp = path_parser.duc / f'duc_{year}.ranked.lines'

        if exists(dump_dp):
            raise  ValueError(f'Remove the target directory to continue: {dump_dp}')
        os.mkdir(dump_dp)

        doc_fp = dump_dp / f'duc_{year}.source'
        query_fp = dump_dp / f'duc_{year}.query'
        meta_fp = dump_dp / f'duc_{year}.meta'
        
        meta_lines = []
        doc_lines = []
        query_lines = []

        cc_lines = io.open(src_fp).readlines()
        n_records = 0
        for line in cc_lines:
            cc_obj = json.loads(line.strip('\n'))
            query = cc_obj['query']['original_concat']
            doc_objs = cc_obj['docs']
            n_records += len(doc_objs)

            meta_line = f'{cc_obj["cid"]}\t{len(doc_objs)}'
            meta_lines.append(meta_line)

            for doc_obj in doc_objs:
                doc_lines.append(' '.join(doc_obj['original_doc']))  # a list of sentences
                query_lines.append(query)
        
        with io.open(meta_fp, 'a') as meta_f:
            meta_f.write('\n'.join(meta_lines))
        
        with io.open(doc_fp, 'a') as doc_f:
            doc_f.write('\n'.join(doc_lines))  

        with io.open(query_fp, 'a') as query_f:
            query_f.write('\n'.join(query_lines))
                
        lines_q = len(io.open(query_fp).readlines())
        lines_d = len(io.open(doc_fp).readlines())
        assert lines_q == lines_d == n_records, f'line_q: {lines_q}, line_d: {lines_d}, n_records: {n_records}'

        print(f'Dump bart-formatted mds data to: {dump_dp}')


class TDQFSFormatter:    
    def __init__(self):
        pass

    def format_as_single_file(self):
        src_fp = path_parser.tdqfs / f'tdqfs.ranked'
        dump_dp = path_parser.tdqfs / f'tdqfs.ranked.lines'

        if exists(dump_dp):
            raise  ValueError(f'Remove the target directory to continue: {dump_dp}')
        os.mkdir(dump_dp)

        doc_fp = dump_dp / f'tdqfs.source'
        query_fp = dump_dp / f'tdqfs.query'
        meta_fp = dump_dp / f'tdqfs.meta'
        
        meta_lines = []
        doc_lines = []
        query_lines = []

        cc_lines = io.open(src_fp).readlines()
        n_records = 0
        for line in cc_lines:
            cc_obj = json.loads(line.strip('\n'))
            query = cc_obj['query']['title']
            doc_objs = cc_obj['docs']
            n_records += len(doc_objs)

            meta_line = f'{cc_obj["cid"]}\t{len(doc_objs)}'
            meta_lines.append(meta_line)

            for doc_obj in doc_objs:
                doc_lines.append(' '.join(doc_obj['original_doc']))  # a list of sentences
                query_lines.append(query)
        
        with io.open(meta_fp, 'a') as meta_f:
            meta_f.write('\n'.join(meta_lines))
        
        with io.open(doc_fp, 'a') as doc_f:
            doc_f.write('\n'.join(doc_lines))

        with io.open(query_fp, 'a') as query_f:
            query_f.write('\n'.join(query_lines))
                
        lines_q = len(io.open(query_fp).readlines())
        lines_d = len(io.open(doc_fp).readlines())
        assert lines_q == lines_d == n_records, f'line_q: {lines_q}, line_d: {lines_d}, n_records: {n_records}'

        print(f'Dump bart-formatted mds data to: {dump_dp}')
    
    def build_expanded_query_file(self):
        dump_dp = path_parser.tdqfs / f'tdqfs.ranked.lines'
        query_fp = dump_dp / f'tdqfs.query'
        query_lines = io.open(query_fp).readlines()

        expand_ns = 1
        guid_name = f'grsum-tdqfsdoc-0.6_cos-0_wan-ns_{expand_ns}'
        guid_fp = path_parser.guidance / guid_name
        guid_lines = io.open(guid_fp).readlines()
        
        dump_fp = dump_dp / f'tdqfs.query.expand_{expand_ns}'
        expanded_queries = []
        with io.open(dump_fp, 'a') as dump_f:
            for ql, gl in zip(query_lines, guid_lines):
                concat = ql.strip('\n') + ' *=- ' + gl.strip('\n')
                expanded_queries.append(concat)
            
            dump_f.write('\n'.join(expanded_queries))


class MargeFormatter:
    def __init__(self):
        self.years = ['2005', '2006', '2007']
        self.marge_root = Path('/home/shiftsum/unilm_in')

    def load_marge_file(self, year):
        cont_sel_fn = f'unilm_in-rr-34_config-25000_iter-query-ir-dial-tf-{year}-top150-local_pos.json'
        cont_sel_fp = self.marge_root / cont_sel_fn
        cid2content = {}
        with io.open(cont_sel_fp) as cont_sel_f:
            for line in cont_sel_f:
                line = line.strip('\n')
                json_obj = json.loads(line)
                cid, content = json_obj['uid'], json_obj['src']
                cid2content[cid] = content
        return cid2content
        
    def format(self, year):
        cid2content = self.load_marge_file(year)

        src_fp = path_parser.duc / f'duc_{year}.ranked'
        dump_dp = path_parser.duc / f'duc_{year}.marge.lines'
        if exists(dump_dp):
            raise  ValueError(f'Remove the target directory to continue: {dump_dp}')
        os.mkdir(dump_dp)

        doc_fp = dump_dp / f'duc_{year}.source'
        query_fp = dump_dp / f'duc_{year}.query'
        meta_fp = dump_dp / f'duc_{year}.meta'

        meta_lines = []
        doc_lines = []
        query_lines = []

        cc_lines = io.open(src_fp).readlines()
        for line in cc_lines:
            cc_obj = json.loads(line.strip('\n'))
            cid = cc_obj["cid"]
            query = cc_obj['query']['original_concat']
            document = cid2content[cid]

            doc_lines.append(document)
            meta_lines.append(cid)
            query_lines.append(query)
        
        with io.open(meta_fp, 'a') as meta_f:
            meta_f.write('\n'.join(meta_lines))
        
        with io.open(doc_fp, 'a') as doc_f:
            doc_f.write('\n'.join(doc_lines))  

        with io.open(query_fp, 'a') as query_f:
            query_f.write('\n'.join(query_lines))

        print(f'Dump bart-formatted mds data to: {dump_dp}')


def format_duc():
    formatter = DUCFormatter()
    for year in formatter.years:
        formatter.format_as_single_file(year)


def format_tdqfs():
    formatter = TDQFSFormatter()
    formatter.format_as_single_file()


def build_expanded_query_file_for_tdqfs():
    formatter = TDQFSFormatter()
    formatter.build_expanded_query_file()


def format_duc_from_marge():
    formatter = MargeFormatter()
    for year in ['2006', '2007']:
        formatter.format(year)


if __name__ == '__main__':
    # format_duc()
    # format_tdqfs()
    build_expanded_query_file_for_tdqfs()
    # format_duc_from_marge()
