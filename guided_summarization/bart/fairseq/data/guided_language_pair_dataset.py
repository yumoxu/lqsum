# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset, prior_utils

logger = logging.getLogger(__name__)


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
    tag_pad_idx=2,
    use_prior=None, max_num_prior_spans=5, max_prior_ratio=0.5
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    z_tokens = merge('z', left_pad=left_pad_source)
    z_tokens = z_tokens.index_select(0, sort_order)
    z_lengths = torch.LongTensor([s['z'].numel() for s in samples])
    n_z_tokens = sum(len(s['z']) for s in samples)

    def merge_z_tags(left_pad, move_eos_to_beginning=False):
        """
            This function is similar to func:merge().
            We didn't use merge() since, for z_tags, 
            we need to specify tag_pad_idx and eos_idx (which is not used).
        """
        return data_utils.collate_tokens(
            [s['z_tags'] for s in samples],
            pad_idx=tag_pad_idx, eos_idx=None, left_pad=left_pad, move_eos_to_beginning=move_eos_to_beginning,
        )
        
    if 'z_tags' in samples[0] and samples[0]['z_tags'] is not None:
        z_tags = merge_z_tags(left_pad=left_pad_source)
        z_tags = z_tags.index_select(0, sort_order)
        # print(f'[collate] z_tags after merge: {z_tags}')
    else:
        z_tags = None

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    
    z_tags_prior = None
    if use_prior:
        z_tags_prior = prior_utils.build_z_tags_prior(z_tags, 
            max_num_spans=max_num_prior_spans, 
            prior_ratio=max_prior_ratio)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'n_z_tokens': n_z_tokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'z_tokens': z_tokens,
            'z_lengths': z_lengths,
            'z_tags': z_tags,
            'z_tags_prior': z_tags_prior,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens


    return batch


class GuidedLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        z=None, z_sizes=None,
        z_tags=None, tag_dict=None, norm_for_special_token='0',
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        use_prior=None, max_num_prior_spans=5, max_prior_ratio=0.5
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.z = z
        self.z_tags = z_tags
        self.tag_dict = tag_dict
        self.norm_for_special_token = norm_for_special_token

        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.z_sizes = z_sizes
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        if not shuffle:
            logger.warning('shuffle is set to False. Check carefully to see if that is what you want')
        
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

        # for injecting prior info
        self.use_prior = use_prior
        self.max_num_prior_spans = max_num_prior_spans
        self.max_prior_ratio = max_prior_ratio
    
    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        z_item = self.z[index]
        z_tag_item = self.z_tags[index] if self.z_tags else None

        
        ic.disable()
        # ic.enable()

        ic(index)
        ic(src_item)
        ic(z_item)

        # if z_tag_item is not None:
        #     assert z_tag_item.size() == z_item.size(), \
        #         f'Incompatible size in sample {index}: z_tag_item: {z_tag_item.size(0)}, z_item: {z_item.size(0)} \
        #         \nz_tag_item: {z_tag_item}\nz_item: {z_item}'

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

            if self.z[index][-1] != bos:
                z_item = torch.cat([torch.LongTensor([bos]), self.z[index]])
                if self.z_tags:
                    z_tag_item = torch.cat([[torch.LongTensor([self.tag_dict.index(self.norm_for_special_token)])], self.z_tags[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
            if self.z[index][-1] == eos:
                z_item = self.z[index][:-1]

                if self.z_tags:
                    z_tag_item = self.z_tags[index][:-1]

        # if z_tag_item is not None:
        #     assert z_tag_item.size() == z_item.size(), \
        #         f'Incompatible size in sample {index}: z_tag_item: {z_tag_item.size(0)}, z_item: {z_item.size(0)} \
        #         \nz_tag_item: {z_tag_item}\nz_item: {z_item}'
        
        # ic.enable()
        if index == 17:
            ic(f'after: {index}')
            ic(index)
            ic(src_item)
            ic(z_item)
        # ic.disable()
        
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'z': z_item,
            'z_tags': z_tag_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            tag_pad_idx=self.tag_dict.pad() if self.tag_dict else None,
            use_prior=self.use_prior, max_num_prior_spans=self.max_num_prior_spans, max_prior_ratio=self.max_prior_ratio
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0, self.z_sizes[index])
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def ordered_indices_debug(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        ic.enable()
        ic(indices[:100])
        ic.disable()
        return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
