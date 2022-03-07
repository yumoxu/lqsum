# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch

from icecream import install
install()

logger = logging.getLogger(__name__)

POSITIVE_LABEL, NEGATIVE_LABEL = 0, 1

# ic.disable()

def get_spans(z_item):
    """
        Get spans from tags. 
        return: a list of (start, end).
        
        TODO: to be tested
    """
    start_indices = []
    end_indices = []
    prev = NEGATIVE_LABEL
    for idx in range(len(z_item)):
        curr = z_item[idx]
        if curr == POSITIVE_LABEL and prev!=POSITIVE_LABEL:  # prev can be either negative or padding
            start_indices.append(idx)

        if curr == NEGATIVE_LABEL and prev==POSITIVE_LABEL:
            end_indices.append(idx)
        
        prev = curr
    
    if len(end_indices) == len(start_indices)-1:  # close the last span
        end_indices.append(len(z_item))
    
    assert len(start_indices) == len(end_indices), f'Incompatible lengths: start: {len(start_indices)}, end: {len(end_indices)}'
    return list(zip(start_indices, end_indices))


def update_status(status, spans, budget):
    """
        Update status of spans based on the current budget.
        Set spans with sizes over the given budget to 0.

        return: an updated status
        
        TODO: to be tested 
    """
    for i in range(len(spans)):
        if status[i] == 0:
            continue

        if (spans[i][1] - spans[i][0]) > budget:
            status[i] = 0
    return status


def sample_index(status):
    """
        Get probability of be chosen for spans.
        status: 1: available; 0: unavailable (used or over-budget)
        
        return: a list of probabilities
    """
    avai_indices = [idx for idx, st in enumerate(status) if st==1]
    if not avai_indices:  # no span is available
        return None

    index = np.random.choice(avai_indices, 1)[0]
    return index


def sample_spans(spans, budget, num_spans):
    """
        sample from spans based on budget (total spans size), and num_spans (num of spans).
        return: sampled spans.

        TODO: to be tested
    """
    span_samples = []

    status = [1] * len(spans)
    status = update_status(status, spans, budget)
    ic('updated status', status)

    while budget > 0 and num_spans > 0:
        _index = sample_index(status)
        if _index is None:  # terminate when no span is available
            break
        
        ic(_index)
        _span = spans[_index]

        span_samples.append(_span)
        span_size = _span[1] - _span[0]
        
        budget -= span_size
        num_spans -= 1
        status[_index] = 0  # set it used
        
        status = update_status(status, spans, budget)
        ic('updated status', status)

    return span_samples

def sample_num_spans(max_num_spans, batch_size):
    """
        TODO: to be tested
    """
    return np.random.choice(list(range(max_num_spans+1)), batch_size).tolist()

    
def build_z_tags_prior(z_tags, max_num_spans=5, prior_ratio=0.5):
    """
        z_tags: LongTensor, B * L
        n_z_tokens: LongTensor, B * 1

        TODO: to be tested
    """ 
    # ic.enable()
    z_tags_prior = torch.ones_like(z_tags)

    num_spans_batch = sample_num_spans(max_num_spans, batch_size=z_tags.size(0))
    ic(num_spans_batch)

    for idx, z_item in enumerate(z_tags):
        budget = (z_item == 0).int().sum().item() * prior_ratio 
        spans = get_spans(z_item)
        ic(budget)
        ic(spans)
        num_spans = num_spans_batch[idx]
        ic(num_spans)
        span_samples = sample_spans(spans, budget, num_spans=num_spans)
        ic(span_samples)
        ic(z_item)
        
        for _span in span_samples:
            _span_size = _span[1] - _span[0]
            z_tags_prior[idx, _span[0]:_span[1]] = torch.LongTensor([POSITIVE_LABEL] * _span_size)  # set it to query tokens
        
        ic(z_tags_prior[idx])
        ic('------------------')

    ic(z_tags_prior)
    ic(z_tags)
    return z_tags_prior


if __name__ == '__main__':
    z_tags = torch.LongTensor([[2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], 
        [2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0], 
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0]])
    
    # z_lengths = torch.LongTensor([[6], [5], [7], [0], [0], [3]])

    build_z_tags_prior(z_tags, max_num_spans=3, prior_ratio=1.0)
