# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from fairseq import utils
from fairseq.data import encoders


logger = logging.getLogger(__name__)


class GuidedBARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def truncate(self, sentence: str):
        tokens = self.bpe.encode(sentence)
        tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        sentences = [tokens.long()]
        #tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        #bpe_sentence = '<s> ' + tokens + ' </s>'
        #sentence = self.bpe.decode(self.task.source_dictionary.string(tokens))
        return sentences[0]

    def encode(self, sentence: str, *addl_sentences, no_separator=True, return_bpe=False) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        
        if return_bpe:
            return tokens.long(), bpe_sentence
        
        return tokens.long()

    def encode_bped_sentence(self, bped_sentence: str, return_bpe=False) -> torch.LongTensor:
        """
            For encoding input in the format of BPE.
        """
        # print('Encode bped sentence')
        if bped_sentence.startswith('<s>'):
            bped_sentence = ' '.join(bped_sentence.split(' ')[1:-1])

        if len(bped_sentence.split(' ')) > self.max_positions - 2:
            bped_sentence = ' '.join(bped_sentence.split(' ')[:self.max_positions - 2])
        
        bped_sentence = '<s> ' + bped_sentence + ' </s>'            

        # print(bped_sentence)
        tokens = self.task.source_dictionary.encode_line(bped_sentence, append_eos=False)
        if return_bpe:
            return tokens.long(), bped_sentence
        
        return tokens.long()

    def encode_bped_sentence_with_tags(self, bped_sentence: str) -> torch.LongTensor:
        """
            For encoding input in the format of BPE with tags.

            Example of bped_sentence:
                112###0 329##1 78###0
        """
        bpe_tag_pairs = [token.split('###') for token in bped_sentence.split()]

        if bpe_tag_pairs[0][0] == '<s>':
            bpe_tag_pairs = bpe_tag_pairs[1:-1]
            
        if len(bpe_tag_pairs) > self.max_positions - 2:
            bpe_tag_pairs = bpe_tag_pairs[:self.max_positions - 2]

        bpes, tags = list(zip(*bpe_tag_pairs))
        bpe_sentence = ' '.join(bpes)
        bpe_sentence = '<s> ' + bpe_sentence + ' </s>'
        
        tags = [int(tag) for tag in tags]
        # when all guidance tokens are masked, a NaN problem will probably arise
        # if only the input tags are used (e.g., will be fine if the estimated tags are also considered)
        # in this case, we can choose to expose all guidance tokens so they are treated equally
        turn_all_zeros_to_all_ones = False
        if turn_all_zeros_to_all_ones and sum(tags) == 0:
            tags = [1] * len(tags)
        
        # tagging can be integrated into the padding mask for joint masking
        # to adapt to the padding mask scheme
        # we set 1: ignore, 0: attend 
        tags = [1-tag for tag in tags]
        bound_tag = 1
        tags = [bound_tag] + tags + [bound_tag]
        
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        tags = torch.IntTensor(tags)
        assert tags.size() == tokens.size(), f'Incompatible size: tags: {tags.size()}, tokens: {tokens.size(0)}'
        # ic.enable()
        # ic(tokens.size())
        # ic.disable()

        return tokens.long(), tags.long()

    def encode_with_tags(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        word_tag_pairs = [token.split('###') for token in sentence.split()]
        
        tokens = ''
        tags = []
        for word, tag in word_tag_pairs:
            _tokens = self.bpe.encode(word)
            if not _tokens:
                continue
            
            if tokens:  # space between words but not for the first word
                tokens += ' '
            tokens += _tokens
            tags.extend([int(tag)] * len(_tokens.split()))
        
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
            tags = tags[:self.max_positions - 2]
        
        
        # when all guidance tokens are masked, there will be a NaN problem
        # in this case, we expose all guidance tokens to treat them equally
        if sum(tags) == 0:
            tags = [1] * len(tags)
        
        # tagging can be integrated into the padding mask for joint masking
        # to adapt to the padding mask scheme
        # we set 1: ignore, 0: attend 
        tags = [1-tag for tag in tags]

        bpe_sentence = '<s> ' + tokens + ' </s>'
        bound_tag = 1
        tags = [bound_tag] + tags + [bound_tag]
        
        assert not addl_sentences, 'Not implemented!'
        
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        tags = torch.IntTensor(tags)

        assert tags.size() == tokens.size(), f'Incompatible size: tags: {tags.size()}, tokens: {tokens.size(0)}'

        return tokens.long(), tags.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        # for s in sentences:  # debug
        #     print(f'sentence: {s}')
        
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor], zs: List[torch.LongTensor], z_tags: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            zs,
            [z.numel() for z in zs],
            z_tags=z_tags,
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self, sentences: List[str], zs: List[str], 
        tag_mode: str, 
        beam: int = 1, 
        use_bped_input: bool = False,
        verbose: bool = False, 
        stddec: bool = False,
        stdenc: bool = False,
        proc_no_match = False,
        **kwargs) -> str:
        """
            tag_mode: str, controls the injection of tags
            use_bped_input: bool, defines whether sentences in zs are byte-pairs
        """
        input = [self.encode(sentence) for sentence in sentences]

        def _enc_z_and_tags(sentence, bped):
            if bped:
                return self.encode_bped_sentence_with_tags(sentence)
            return self.encode_with_tags(sentence)

        if tag_mode:
            pairs = [_enc_z_and_tags(sentence, bped=use_bped_input) 
                for sentence in zs]
            z, z_tags = list(zip(*pairs))
        else:
            z = [self.encode(sentence) for sentence in zs]
            z_tags = None
        
        if proc_no_match:
            """
                When proc_no_match is True, we use z as input.
                This is to make sure input and z_tags are consistent, 
                as when there is no query in the input, we form z as the concatenation of input and query.
            """
            input = z
        
        hypos = self.generate(input, z, z_tags, tag_mode, beam, verbose, stddec=stddec, stdenc=stdenc, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def get_tokens(self, tensor, escape_unk=False):
        """
            yumo's revision: help to convert token ids to actual tokens, with help of source_dict.

        """
        # dictionary = self.task.source_dictionary
        dictionary = self.task.source_dictionary

        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return [self.get_tokens(t, escape_unk) for t in tensor]

        def token_string(i):
            if i == dictionary.unk():
                return dictionary.unk_string(escape_unk)
            else:
                return dictionary[i]

        if hasattr(dictionary, "bos_index"):
            token_ids = [token_string(i)
                for i in tensor
                if (i != dictionary.eos()) and (i != dictionary.bos())]
        else:
            token_ids = [token_string(i) for i in tensor if i != dictionary.eos()]
        
        n_pads = 0
        tokens = []
        new_token_ids = []
        for s in token_ids:
            if s == '<pad>':
                n_pads += 1
                continue
            tokens.append(self.bpe.decode(str(s)))
            new_token_ids.append(s)
        
        return new_token_ids, tokens, n_pads

    def tag(self, sentences: List[str], zs: List[str], use_bped_input: bool):
        """
            Added by yumo: get tagging scores for tokens
        """ 
        src, src_bpe = [], []
        for sentence in sentences:
            if use_bped_input:
                tokens, bpe = self.encode_bped_sentence(sentence, return_bpe=True)
            else:
                tokens, bpe = self.encode(sentence, return_bpe=True)
            
            src.append(tokens)
            src_bpe.append(bpe)

        z, z_bpe = [], []
        for sentence in zs:
            if use_bped_input:
                tokens, bpe = self.encode_bped_sentence(sentence, return_bpe=True)
            else:
                tokens, bpe = self.encode(sentence, return_bpe=True)
            
            z.append(tokens)
            z_bpe.append(bpe)
        
        sample = self._build_sample(src, z, z_tags=None)

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        # copied from sequence_generator.py
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        # compute the encoder output for each beam
        with torch.no_grad():
            tag_scores = self.model.z_forward_encoder(**encoder_input, tag_only=True)

        # re-order inputs and outputs 
        id = sample['id'].tolist()
        # print(f'id: {id}')
        tag_scores = torch.FloatTensor([v for _, v in sorted(zip(id, tag_scores.tolist()))])
        src_token_ids = torch.IntTensor([v for _, v in sorted(zip(id, sample['net_input']['src_tokens'].tolist()))])
        z_token_ids = torch.IntTensor([v for _, v in sorted(zip(id, sample['net_input']['z_tokens'].tolist()))])

        src_tokens = self.get_tokens(src_token_ids)
        z_tokens = self.get_tokens(z_token_ids)
        # z_tokens = self.task.target_dictionary.string(z_token_ids)
        # print(f'[guided_hub_interface] z_token_ids: {z_token_ids}')
        # print(f'[guided_hub_interface] z_tokens: {z_tokens}')
        
        return {
            'tag_scores': tag_scores,
            'src_token_ids': src_token_ids,
            'z_token_ids': z_token_ids,
            'src_tokens': src_tokens,
            'z_tokens': z_tokens,
            'z_bpe': z_bpe,
            # 'z_n_pad': z_n_pad,
        }

    def generate(self, tokens: List[torch.LongTensor], zs: List[torch.LongTensor], z_tags: List[torch.LongTensor],
            tag_mode: str = '', 
            beam: int = 5, verbose: bool = False, 
            stddec: bool = False,
            stdenc: bool = False,
            **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens, zs, z_tags=z_tags)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        gen_args.tag_mode = tag_mode  # add tag_mode into args for generation
        gen_args.stddec = stddec 
        gen_args.stdenc = stdenc

        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
