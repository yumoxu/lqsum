import torch
from fairseq.models.bart import GuidedBARTModel
from torch import argmax
import torch.nn.functional as F

from tqdm import tqdm

from icecream import install
install()

import sys
bart = GuidedBARTModel.from_pretrained(
    sys.argv[4],
    checkpoint_file=sys.argv[5],
    data_name_or_path=sys.argv[6]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 16

if 'cnndm' in sys.argv[6]:
    testset = 'cnndm'
    test_size = 11490
elif 'wikiref' in sys.argv[6]:
    testset = 'wikiref'
    test_size = 12000
else:
    raise ValueError(f'Test set unrecognized from the path: {sys.argv[6]}')

print(f'testset: {testset}')

TAGGER_EVAL_MODE = True
if TAGGER_EVAL_MODE:
    REMOVE_START_END, TAG_ONLY = False, True
else:
    REMOVE_START_END, TAG_ONLY = True, False

print(f'TAGGER_EVAL_MODE: {TAGGER_EVAL_MODE}')


with open(sys.argv[1]) as source, open(sys.argv[2]) as zs, open(sys.argv[3], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    zline = zs.readline().strip()
    zlines = [zline]

    for sline, zline in tqdm(zip(source, zs), total=test_size):
        if count % bsz == 0:
            with torch.no_grad(): 
                # tag_scores, input_bpe, zs_bpe = bart.tag(slines, zlines)
                tag_res = bart.tag(slines, zlines)
                tag_scores = tag_res['tag_scores']
                src_token_ids = tag_res['src_token_ids']
                z_token_ids = tag_res['z_token_ids']
                src_tokens = tag_res['src_tokens']
                z_tokens = tag_res['z_tokens']
                z_bpe = tag_res['z_bpe']

            # ic(tag_scores.size())
            # ic(z_token_ids.size())

            z_token_ids = z_token_ids.tolist()
            tag_pred = get_tags(tag_scores)
            prob_pred = get_probs(tag_scores)
            
            # ic(tag_scores)
            # ic(tag_pred)
            # ic(z_tokens)
            
            assert len(z_tokens) == len(tag_pred), f'z_tokens: {len(z_tokens)}, tag_pred: {len(tag_pred)}'
            for i in range(len(z_tokens)):
                if TAGGER_EVAL_MODE:
                    _ = build_json_output_for_eval(z_tokens=z_tokens, 
                        tag_pred=tag_pred, 
                        prob_pred=prob_pred,
                        i=i, 
                        fout=fout)
                else:
                    rec = build_output(z_token_ids=z_token_ids, 
                        z_bpe=z_bpe,
                        tag_pred=tag_pred, 
                        i=i, 
                        fout=fout,
                        remove_start_end=REMOVE_START_END, 
                        tag_only=TAG_ONLY)

                fout.flush()
                # assert False

            slines = []
            zlines = []

        slines.append(sline.strip())
        zlines.append(zline.strip())
        count += 1
    
    if slines != []:
        assert zlines
        with torch.no_grad(): 
            # tag_scores, input_bpe, zs_bpe = bart.tag(slines, zlines)
            tag_res = bart.tag(slines, zlines)
            tag_scores = tag_res['tag_scores']
            src_token_ids = tag_res['src_token_ids']
            z_token_ids = tag_res['z_token_ids']
            src_tokens = tag_res['src_tokens']
            z_tokens = tag_res['z_tokens']
            z_bpe = tag_res['z_bpe']

        # ic(tag_scores.size())
        # ic(z_token_ids.size())

        z_token_ids = z_token_ids.tolist()
        tag_pred = get_tags(tag_scores)
        prob_pred = get_probs(tag_scores)
        
        assert len(z_tokens) == len(tag_pred), f'z_tokens: {len(z_tokens)}, tag_pred: {len(tag_pred)}'
        for i in range(len(z_tokens)):
            if TAGGER_EVAL_MODE:
                _ = build_json_output_for_eval(z_tokens=z_tokens, 
                    tag_pred=tag_pred, 
                    prob_pred=prob_pred,
                    i=i, 
                    fout=fout)
            else:
                rec = build_output(z_token_ids=z_token_ids, 
                    z_bpe=z_bpe,
                    tag_pred=tag_pred, 
                    i=i, 
                    fout=fout,
                    remove_start_end=REMOVE_START_END, 
                    tag_only=TAG_ONLY)
            
            fout.flush()
