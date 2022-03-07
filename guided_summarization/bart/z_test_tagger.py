import torch
from fairseq.models.bart import GuidedBARTModel
from torch import argmax
import torch.nn.functional as F

from tqdm import tqdm
import json

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
    USE_BPED_INPUT = True
else:
    REMOVE_START_END, TAG_ONLY = True, False
    USE_BPED_INPUT = False

print(f'TAGGER_EVAL_MODE: {TAGGER_EVAL_MODE}')

from pathlib import Path
TEST_BPE_FILE = Path('/home/lqsum/data/cnndm_guide_source/with_tags/test.source.bu/test.source.bu.bpe')


def build_output(z_token_ids, z_bpe, tag_pred, i, fout, remove_start_end=False, tag_only=False):
    START, PAD, END = '0', '1', '2'
    tokens, tags = z_token_ids[i], tag_pred[i]

    bpes = z_bpe[i].split()
    if remove_start_end:
        bpes = bpes[1:-1]

    ic(len(bpes))
    assert len(tags) == len(tokens), f'#tokens: {len(tokens)}, #tags: {len(tags)}'

    items = []
    cur = 0
    n_pads = 0

    ic(len(tokens))
    for i in range(len(tokens)):
        _token, _tag = str(tokens[i]), str(tags[i])
        if _token == PAD:  # pad index
            n_pads += 1
            continue
        
        if remove_start_end and _token in (START, END):
            continue
        
        assert cur < len(bpes), f'cur: {cur}, i: {i}, n_pads: {n_pads}'
        item = _tag if tag_only else bpes[cur] + '###' + _tag
        items.append(item)
        cur += 1
    
    assert cur == len(bpes), f'cur: {cur}, #bpe: {len(bpes)}'

    # untagged_tokens = _tokens[n_items:]
    # if untagged_tokens:
    #     for token in untagged_tokens:
    #         item = token + '###' + '1'
    #         items.append(item)

    rec = ' '.join(items)
    fout.write(rec + '\n')
    return rec

    
def sanity_check(token_ids, bpeline):
    """
        validate the ids with ids in the test file
        this ensures the consistency between ref tags and predicted tags

    """
    ref_ids = bpeline.split()[:len(token_ids)]
    assert token_ids == ref_ids, f'token_ids: {token_ids}, bpe_ids: {ref_ids}'
    # for idx in range(len(token_ids)):
    #     if token_ids[idx] != ref_ids[idx]:
    #         assert False, f'idx: {idx}, token_ids: {token_ids[idx]}, bpe_ids: {ref_ids[idx]}'
            # if token_ids[idx] == '220':
            #     del token_ids[idx]
            #     del tokens[idx]
            #     del probs[idx]
            #     del tags[idx]


def build_json_output_for_eval(z_tokens, tag_pred, prob_pred, i, bpelines, fout):
    token_ids, tokens, n_pads = z_tokens[i]
    tags, probs = tag_pred[i], prob_pred[i]

    # remove bos and eos for tags and probs
    # do not need to apply it on tokens, as this has been handled when converting from ids
    tags = tags[1:-1]
    probs = probs[1:-1]

    assert len(tags) == len(tokens)+n_pads, f'#tokens: {len(tokens)}, n_pads: {n_pads}, #tags: {len(tags)}'

    probs = [[prob[1], prob[0]] for prob in probs[n_pads:]]  # swap indices: index 1 should indicate importance
    tags = [1-item for item in tags[n_pads:]]  # for eval purpose, the scheme is 1 for selection

    ic(sum(tags))

    tags = [str(tag) for tag in tags]
    # print(f'tokens after pad removal: {tokens}')

    sanity_check(token_ids, bpeline=bpelines[i])

    assert len(tokens) == len(probs) == len(tags), f'#tokens: {len(tokens)}, #probs: {len(probs)}, #tags: {len(tags)}'

    json_data = {
        "words": tokens,
        "class_probabilities": probs,
        "tags": tags,
        "bpe_ids": token_ids,
    }
    json.dump(json_data, fout)
    fout.write('\n')
    return json_data


def get_tags(tag_scores):
    return argmax(tag_scores, dim=-1).tolist()  # B * T, 0: to reserve, 1: to mask


def get_probs(tag_scores):
    tag_prob = F.softmax(tag_scores, dim=-1)
    return tag_prob.tolist()
    # return tag_prob[:, :, 0].tolist()  # B * T, 0: to reserve, 1: to mask


with open(sys.argv[1]) as source, open(sys.argv[2]) as zs, open(sys.argv[3], 'w') as fout, open(TEST_BPE_FILE) as bpe_f:
    sline = source.readline().strip()
    slines = [sline]
    zline = zs.readline().strip()
    zlines = [zline]

    bpeline = bpe_f.readline().strip()
    bpelines = [bpeline]
    # print(bpelines)

    for sline, zline, bpeline in tqdm(zip(source, zs, bpe_f), total=test_size):
        if count % bsz == 0:
            with torch.no_grad():
                # ic(slines)
                # ic(zlines)
                tag_res = bart.tag(slines, zlines, use_bped_input=USE_BPED_INPUT)
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
                        bpelines=bpelines,
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
            bpelines = []

        slines.append(sline.strip())
        zlines.append(zline.strip())
        bpelines.append(bpeline.strip())
        count += 1
    
    if slines != []:
        assert zlines
        assert bpelines

        with torch.no_grad(): 
            tag_res = bart.tag(slines, zlines, use_bped_input=USE_BPED_INPUT)
            tag_scores = tag_res['tag_scores']
            src_token_ids = tag_res['src_token_ids']
            z_token_ids = tag_res['z_token_ids']
            src_tokens = tag_res['src_tokens']
            z_tokens = tag_res['z_tokens']
            z_bpe = tag_res['z_bpe']

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
                    bpelines=bpelines,
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
