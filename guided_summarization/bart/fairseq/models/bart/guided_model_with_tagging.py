# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import logging

import torch.nn as nn
import torch

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.guided_transformer import (
    GuidedTransformerModelwithTagging, 
    GuidedTransformerModelwithTaggingAndStdDec,
    GuidedTransformerModelwithTaggingAndStdEncDec,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_encoder import EncoderOut
import torch.nn.functional as F

from .guided_hub_interface import GuidedBARTHubInterface
#from .hub_interface import BARTHubInterface
from fairseq.modules import (
    AdaptiveSoftmax,
    GradMultiply,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    GuidedTransformerEncoderLayer,
    GuidedTransformerDecoderLayer,
    NasTransformerDecoderLayer,
)
from torch import Tensor, argmax, logical_or
import numpy as np

logger = logging.getLogger(__name__)


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, dim: int = -1, inject_noise: bool = True):
    """
        Samples from the Gumbel-Softmax distribution and optionally discretizes.

        Args:
            logits: `[..., num_features]` unnormalized log probabilities
            tau: non-negative scalar temperature
            hard: if ``True``, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int): A dimension along which softmax will be computed. Default: -1.
            inject_noise (bool): One can choose to not use Gumbel noise, usually in validation and testing. Default: True.

        Returns:
            Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
            If ``hard=True``, the returned samples will be one-hot, otherwise they will
            be probability distributions that sum to 1 across `dim`.
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():  # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    if inject_noise:
        # gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0,1)
        gumbels = _gen_gumbels()  # ~Gumbel(0,1)
        logits = logits + gumbels
    
    gumbels = logits / tau  # ~Gumbel(logits,tau) if inject_noise

    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


@register_model('guided_bart_with_tagging')
class GuidedBARTModelwithTagging(GuidedTransformerModelwithTagging):
    """
        This class is the same as `GuidedBARTModelOriginal`, 
        except that it allows `z_tags` as an optional input in forward.
    """

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def name(self):
        return 'guided_bart_with_tagging'

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

        # yumo's revision: add a tagging head
        if args.tagging_head_name:
            self.register_classification_head(name=args.tagging_head_name, num_classes=2, inner_dim=args.encoder_embed_dim)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.mask_z_tokens_with_tags = args.mask_z_tokens_with_tags
        self.mask_z_tokens_with_scores = args.mask_z_tokens_with_scores
        self.mask_z_tokens_with_gumbel_scores = args.mask_z_tokens_with_gumbel_scores

        self.mask_score_threshold = args.mask_score_threshold
        self.mask_z_tokens_with_straight_through = args.mask_z_tokens_with_straight_through
        self.mask_z_tokens_with_gumbel_straight_through = args.mask_z_tokens_with_gumbel_straight_through

        if self.mask_z_tokens_with_tags:
            logger.info(f'mask_z_tokens_with_tags: {self.mask_z_tokens_with_tags}')

        if self.mask_z_tokens_with_scores:
            logger.info(f'mask_z_tokens_with_scores: {self.mask_z_tokens_with_scores}')
            logger.info(f'mask_score_threshold: {self.mask_score_threshold}')

        if self.mask_z_tokens_with_straight_through:
            logger.info(f'mask_z_tokens_with_straight_through: {self.mask_z_tokens_with_straight_through}')

        if self.mask_z_tokens_with_gumbel_straight_through:
            logger.info(f'mask_z_tokens_with_gumbel_straight_through: {self.mask_z_tokens_with_gumbel_straight_through}')

        if self.mask_z_tokens_with_gumbel_scores:
            logger.info(f'mask_z_tokens_with_gumbel_scores: {self.mask_z_tokens_with_gumbel_scores}')
        
        self.tag_on_doc = args.tag_on_doc
        self.use_one_encoder = args.use_one_encoder
        self.apply_mask_to_shared_enc_layer = args.apply_mask_to_shared_enc_layer

        self.query_model_lr_multiplier = args.query_model_lr_multiplier

        # tau is an attribute in args which is dynamically set during training
        # so self.tau == None during training
        # during training/validation, we feed tau via batch (see train.py), and also update it in args
        # during tesing, we have arg.tau and use it as self.tau
        self.tau = args.tau if hasattr(args, 'tau') else None
        logger.info(f'Set self.tau to: {self.tau}')
    
    @staticmethod
    def add_args(parser):
        super(GuidedBARTModelwithTagging, GuidedBARTModelwithTagging).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )

        parser.add_argument(
            '--tagging-head-name',
            help='head name for tagging'
        )

        parser.add_argument(
            '--mask-z-tokens-with-tags', action="store_true", default=False,
            help='prevent decoding from attending tokens classified as necessary'
        )

        parser.add_argument(
            '--mask-z-tokens-with-scores', action="store_true", default=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--mask-z-tokens-with-gumbel-scores', action="store_true", default=False,
            help='weigh representations with their salience scores injected with gumbel noise'
        )

        parser.add_argument(
            '--mask-z-tokens-with-straight-through', action="store_true", default=False,
            help='hard and differentiable masking with straight through trick'
        )

        parser.add_argument(
            '--mask-z-tokens-with-gumbel-straight-through', action="store_true", default=False,
            help='hard and differentiable masking with sampling and straight through trick'
        )
        
        parser.add_argument(
            '--mask-score-threshold', type=float, metavar='D', default=-1.0,
            required=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--query-model-lr-multiplier', type=float, metavar='D', default=1.0,
            required=False,
            help='increase the learning rate by this multiplier for params in the query model'
        )

        parser.add_argument(
            '--fixed-oracle-tag-prob', type=float, metavar='D',
            required=False,
            help='probability of using oracle tags in updating decoder mask, when mask-z-tokens==True'
        )

        parser.add_argument(
            '--warmup-updates-with-oracle-tags', type=int, metavar='D', default=1,
            required=False,
            help='for the first specified steps, use oracle tags for training.'
        )

        parser.add_argument(
            '--oracle-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do oracle probability annealing till this step.'
        )

        parser.add_argument(
            '--oracle-anneal-end-value', type=float, metavar='D',
            required=False,
            help='Do oracle probability annealing till this value.'
        )

        parser.add_argument(
            '--tau-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do tau annealing till this step.'
        )

        parser.add_argument(
            '--tag-on-doc', action="store_true", default=False,
            help='tag on the original document. When set to True, we produced shared representations for x and z, via encoding z.'
        )

        parser.add_argument(
            '--use-one-encoder', action="store_true", default=False,
            help='use only when encoder for x and z. Only allowed when tag-on-doc is used. The last layer in encoder for z is not used.'
        )

        parser.add_argument(
            '--apply-mask-to-shared-enc-layer', action="store_true", default=False,
            help='throw away the last layer of z_encoder and use the second last layer for decoding.'
        )

        parser.add_argument(
            '--fixed-tau', type=float, metavar='D',
            required=False,
            help='Constant tau for temperature controlling in Gumbel Softmax.'
        )

        parser.add_argument(
            '--tau', type=float, metavar='D',
            required=False,
            help='Dynamic tau for temperature controlling in Gumbel Softmax.'
        )

        parser.add_argument(
            '--use-prior', action="store_true", default=False,
            help='use prior query information, sampled or given.'
        )

        parser.add_argument(
            '--max-num-prior-spans', type=int, metavar='D',
            default=None,
            required=False,
            help='the max number of prior spans allowed to be sampled for query prior.'
        )

        parser.add_argument(
            '--max-prior-ratio', type=float, metavar='D',
            default=None,
            required=False,
            help='the max number of tokens allowed for query prior, as a ratio of total number of query tokens.'
        )
    
    def get_tags(self, sample):
        """Get tags and their lengths from either the sample"""
        return sample['net_input']['z_tags'], sample['net_input']["z_lengths"]
    
    @property
    def supported_targets(self):
        return {'self'}

    def z_forward_encoder(self,
        src_tokens, src_lengths,
        features_only=False, classification_head_name=None, 
        z_tokens=None, z_lengths=None, z_tags=None, tag_mode='',
        tag_only=None,
        inject_noise=False,
        **kwargs
    ):
        """
            For generation.
            z_tags: oracle tags
            tag_mode: 
                - '': oracle tags not applied
                - has `query` or `q`: z_tags contains query tags which is integrated with estimated tags
        """
        if self.tag_on_doc:
            # we tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 768
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            # logger.info(f'src_tokens: {src_tokens}')
            # logger.info(f'z_tokens: {z_tokens}')
            # logger.info('---------------------------')
            # ic(self.tag_on_doc)
            src_tokens, src_lengths = z_tokens, z_lengths
        
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens = True if self.tag_on_doc else False,
            **kwargs,
        )

        # if z_tokens is not None:
            # assert z_tags is not None, 'z_tags is not a part of inputs'

        cached_states, cached_embeddings, cached_mask = None, None, None
        if self.tag_on_doc:
            # states in the last layer is for x
            # and states in the second last layer is for share
            cached_states = encoder_out.encoder_states[-2]  
            cached_embeddings = encoder_out.encoder_embedding
            cached_mask = encoder_out.encoder_padding_mask

        # you can also use z_tags to mask inside self.encoder by setting tag_model
        # but it is just for model analysis
        # at inference, z_tags should only be used for oracle results
        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            z_tags=z_tags,
            tag_mode=tag_mode,
            return_all_hiddens=True,
            cached_states=cached_states,
            cached_embeddings=cached_embeddings,
            cached_mask=cached_mask,
            **kwargs,
        )
        
        def _use_query_tags(tag_mode, default_query_coef=0.0):
            """
                tag_mode: query_1.0
            """
            if 'query' in tag_mode or 'q' in tag_mode:
                tag_items = tag_mode.split('_')
                if len(tag_items) >= 2:
                    query_coef = float(tag_items[-1])
                    return True, query_coef
                return True, default_query_coef
            
            return False, default_query_coef

        ic.enable()
        
        use_query_tags, query_coef = _use_query_tags(tag_mode, default_query_coef=0.0)
        ic(use_query_tags)
        ic(query_coef)
        # when oracle_tag_prob is set to 1.0, 
        # the input z_tags will utterly replace the estimated tags
        # which should be only used when z_tags are oracle tags
        # otherwise, e.g., z_tags are annotated by query, 
        # we integrate z_tags and estimated tags
        oracle_tag_prob = 0.0
        if tag_mode and not use_query_tags: # when tag_mode is set, oracle tags are applied
            oracle_tag_prob = 1.0
        # tau = 0.01

        # ic(tag_mode)
        # ic(oracle_tag_prob)
        # ic(tau)
        # inject_noise = True  # this line is for debugging; should be commented out

        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            # ic(tag_scores)
            if tag_only:
                # print('return tag scores')
                return tag_scores
            
            if self.mask_z_tokens_with_tags:
                # ic(self.mask_z_tokens_with_tags)
                z_encoder_out = self.update_decoding_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob,
                    use_query_tags=use_query_tags, query_coef=query_coef)

            # TODO: add parameters `use_query_tags` and `query_coef` to the following functions:
            # TODO: soft_mask, straight_through, gumbel_straight_through
            if self.mask_z_tokens_with_scores:
                # ic(self.mask_z_tokens_with_scores)
                z_encoder_out = self.soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

            if self.mask_z_tokens_with_gumbel_scores:
                ic(self.tau)
                assert self.tau is not None  # use saved tau in the checkpoint for testing
                z_encoder_out = self.gumbel_soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=self.tau, inject_noise=inject_noise,
                    use_query_tags=use_query_tags, query_coef=query_coef)

            if self.mask_z_tokens_with_straight_through:
                # ic(self.mask_z_tokens_with_straight_through)
                z_encoder_out = self.straight_through(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob)

            if self.mask_z_tokens_with_gumbel_straight_through:
                assert self.tau is not None  # use saved tau in the checkpoint for testing
                z_encoder_out = self.gumbel_straight_through(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=self.tau, inject_noise=inject_noise)

            qt_ratio = self.cal_query_token_ratio(tag_scores, z_encoder_out)

        ic.disable()
        return encoder_out, z_encoder_out

    def z_forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        z_encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        decoder_out = self.decoder(prev_output_tokens, encoder_out, z_encoder_out=z_encoder_out, incremental_state=incremental_state, **extra_args)

        return decoder_out

    def update_decoding_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob, use_query_tags, query_coef):
        """
            tag_scores: B * T * 2

            update encoder_padding_mask with tag_pred back to h_z
            encoder_padding_mask: 1 1 1 1 1 | 0 0 0 0 0
            tag_pred/z_tags: 1 0 1 0 1 | 1 0 1 0 1 (the first 5 should be ignored; attend 7th and 9th tokens)
            results: 1 1 1 1 1 | 1 0 1 0 1 (the first 5 are paddings; attend 7th and 9th tokens))

        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob

        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        def _get_estimated():
            tag_pred = argmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            # ic(tag_pred.size())
            # from copy import copy
            # tag_pred_for_analysis = copy(tag_pred)
            # tag_pred_for_analysis[z_encoder_out.encoder_padding_mask==True] = 2
            # num_zeros = (tag_pred_for_analysis == 0).int().sum().item() / float(tag_pred.size(0))
            # num_ones = (tag_pred_for_analysis == 1).int().sum().item() / float(tag_pred.size(0))
            # zero_ratio = num_zeros/(num_zeros+num_ones)
            # ic(f'keywords: {num_zeros}, masks: {num_ones}, ratio of keywords: {zero_ratio}')
            return logical_or(z_encoder_out.encoder_padding_mask, tag_pred.bool())

        def _get_oracle():
            return logical_or(z_encoder_out.encoder_padding_mask, z_tags.bool())

        if oracle_tag_prob == 1.0:
            updated_mask = _get_oracle()
        elif oracle_tag_prob == 0.0:
            updated_mask = _get_estimated()
        else:
            odd = np.random.rand()
            updated_mask = _get_oracle() if odd <= oracle_tag_prob else _get_estimated()

        def _inject_query_prior(hard_mask, z_tags, query_coef):
            """
                hard_mask: B * T 
                z_tags: B * T
                query_coef: float
                    - 11: set to 1.0
                return: updated hard_mask
            """
            if query_coef == 0.0:
                return hard_mask
            
            if 0.0 < query_coef <= 1.0 or query_coef == 10:  # linear interpolation
                raise ValueError(f'Cannot set query_coef: {query_coef} for hard mask!')
            
            if query_coef == 11:  # set to 1.0
                query_tags = (1.0 - z_tags).bool()  # query_tags now represent salience
                hard_mask = torch.where(~query_tags, query_tags, hard_mask)  # when salient (1), set to 0 (not mask)
                hard_mask = logical_or(z_encoder_out.encoder_padding_mask, hard_mask)  # no change to paddings
                return hard_mask
        
        if use_query_tags:
            updated_mask = _inject_query_prior(updated_mask, z_tags, query_coef)

        def _fix_all_mask_case(updated_mask):
            """
                it is possible that all estimations are zero, which will cause NaN problems.
                in this case, we just use the original mask, which masks nothing but pads.
            """
            all_mask_indices = torch.sum(updated_mask, dim=1) == updated_mask.size(1)

            if all_mask_indices.any():
                n_indices = all_mask_indices.sum()
                logger.info(f'All tokens are masked in {n_indices} samples. Keep the original mask.')
                updated_mask[all_mask_indices] = z_encoder_out.encoder_padding_mask[all_mask_indices]
            
            return updated_mask

        updated_mask = _fix_all_mask_case(updated_mask)
        
        return z_encoder_out._replace(encoder_padding_mask=updated_mask)
        
    def straight_through(self, z_encoder_out, tag_scores, z_tags, oracle_tag_prob, return_mask=None):
        # ic.enable()

        odd = np.random.rand()
        # ic(odd)
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            hard_mask = (1.0 - z_tags).half()
            hard_mask = torch.unsqueeze(hard_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            dim = -1
            y_soft = F.softmax(tag_scores, dim=dim)  # B * T * 2, 0: to reserve, 1: to mask

            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)  # keep max and set it to 1
            one_hot = y_hard - y_soft.detach() + y_soft
            # ic(y_soft)
            # ic(one_hot)

            hard_mask = one_hot[:, :, 0:1]


        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = hard_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        # ic.enable()
        # ic(z_encoder_out.encoder_out.size())
        # ic(soft_mask.size())
        # ic(updated_encoder_out.size())
        # ic.disable()
        if return_mask:
            return z_encoder_out._replace(encoder_out=updated_encoder_out), hard_mask
        return z_encoder_out._replace(encoder_out=updated_encoder_out)
    
    def gumbel_straight_through(self, z_encoder_out, tag_scores, z_tags, oracle_tag_prob, 
            tau, inject_noise, return_mask=None):
        # ic.enable()
        assert tau, 'specify tau when in gumbel_softmax!'

        odd = np.random.rand()
        # ic(odd)
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            hard_mask = (1.0 - z_tags).half()
            hard_mask = torch.unsqueeze(hard_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            one_hot = gumbel_softmax(logits=tag_scores, tau=tau, hard=True, inject_noise=inject_noise)

            hard_mask = one_hot[:, :, 0:1]

        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = hard_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        # ic.enable()
        # ic(z_encoder_out.encoder_out.size())
        # ic(soft_mask.size())
        # ic(updated_encoder_out.size())
        # ic.disable()
        if return_mask:
            return z_encoder_out._replace(encoder_out=updated_encoder_out), hard_mask
        return z_encoder_out._replace(encoder_out=updated_encoder_out)
    
    def soft_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob, threshold):
        """
            tag_scores: B * T * 2
            z_tags: B * T
        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob
        
        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        
        odd = np.random.rand()
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            soft_mask = (1.0 - z_tags).half()
            soft_mask = torch.unsqueeze(soft_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            tag_dist = F.softmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            soft_mask = tag_dist[:, :, 0:1]
            if threshold > 0:
                soft_mask = (soft_mask>=threshold).float().half()

                # n_ones = torch.sum(torch.squeeze(soft_mask, dim=-1), dim=-1)
                ic.enable()
                ic(threshold)
                # ic(n_ones)
                ic.disable()

        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = soft_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        # ic.enable()
        # ic(z_encoder_out.encoder_out.size())
        # ic(soft_mask.size())
        # ic(updated_encoder_out.size())
        # ic.disable()
        
        return z_encoder_out._replace(encoder_out=updated_encoder_out)
    
    def gumbel_soft_mask(self, z_encoder_out, tag_scores, z_tags, oracle_tag_prob, tau, inject_noise, 
            use_query_tags, query_coef):
        ic.enable()
        assert tau, 'specify tau when in gumbel_softmax!'

        odd = np.random.rand()
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            soft_mask = (1.0 - z_tags).half()
            soft_mask = torch.unsqueeze(soft_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            tag_dist = gumbel_softmax(logits=tag_scores, tau=tau, hard=False, inject_noise=inject_noise)
            soft_mask = tag_dist[:, :, 0:1]  # B * T * 1

        def _inject_query_prior(soft_mask, z_tags, query_coef, padding_mask):
            """
                Incorporate prior query information from `z_tags` to `soft_mask`, 
                controlled via query_coef.

                soft_mask: B * T * 1
                z_tags: B * T
                query_coef: float
                    - [0.0, 1.0]: linear interpolation
                    - 10: set to max
                    - 11: set to 1.0
                padding_mask: B * T

                return: updated soft_mask
            """
            # ic(query_tags.size())
            # ic(soft_mask.size())
            if 0.0 <= query_coef <= 1.0:  # linear interpolation
                query_tags = (1.0 - z_tags).half()
                query_tags = torch.unsqueeze(query_tags, dim=-1)
                soft_mask = query_coef * query_tags + (1-query_coef) * soft_mask
                return soft_mask
            
            if query_coef == 10:  # set to max
                query_tags = (1.0 - z_tags).bool()
                # ic(padding_mask.size())
                soft_mask = soft_mask.squeeze(-1)  # B * T
                soft_mask.masked_fill_(padding_mask, 0.)  # B * T
                
                max_values = torch.max(soft_mask, dim=-1, keepdim=False)[0]  # B,
                # ic(max_values)
                seq_len = padding_mask.size(-1)
                max_values = torch.stack([max_values]*seq_len, dim=-1)  # B * T
                # ic(max_values.size())

                soft_mask = torch.where(query_tags, max_values, soft_mask)  # set query tokens to max_values; B * T
                soft_mask = soft_mask.unsqueeze(-1)  # B * T * 1
                return soft_mask
            
            if query_coef == 11:  # set to 1.0
                query_tags = (1.0 - z_tags).bool()
                soft_mask = soft_mask.squeeze(-1)  # B * T
                soft_mask = torch.where(query_tags, query_tags.half(), soft_mask)  # set query tokens to 1.0; B * T
                soft_mask = soft_mask.unsqueeze(-1)  # B * T * 1
                return soft_mask
        
        if use_query_tags:
            soft_mask = _inject_query_prior(soft_mask, z_tags, query_coef, 
                padding_mask=z_encoder_out.encoder_padding_mask)
        
        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = soft_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        ic.disable()
        return z_encoder_out._replace(encoder_out=updated_encoder_out)

    def cal_query_token_ratio(self, tag_scores, z_encoder_out):
        """
            Measure the ratio of tokens deemed as "query tokens" (keep prob>0.5) from tag_scores.
        
            tag_scores: B * T * 2

            update encoder_padding_mask with tag_pred back to h_z
            encoder_padding_mask: 1 1 1 1 1 | 0 0 0 0 0
            tag_pred/z_tags: 1 0 1 0 1 | 1 0 1 0 1 (the first 5 should be ignored; attend 7th and 9th tokens)
            results: 1 1 1 1 1 | 1 0 1 0 1 (the first 5 are paddings; attend 7th and 9th tokens))
        
        """
        tag_pred = argmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
        # ic(tag_pred.size())
        # from copy import copy
        # tag_pred_for_analysis = copy(tag_pred)
        # tag_pred_for_analysis[z_encoder_out.encoder_padding_mask==True] = 2
        # num_zeros = (tag_pred_for_analysis == 0).int().sum().item() / float(tag_pred.size(0))
        # num_ones = (tag_pred_for_analysis == 1).int().sum().item() / float(tag_pred.size(0))
        # zero_ratio = num_zeros/(num_zeros+num_ones)
        mask = z_encoder_out.encoder_padding_mask
        batch_size = mask.size(0)

        num_tokens = (mask == 0).int().sum().item() / batch_size
        
        new_mask = logical_or(z_encoder_out.encoder_padding_mask, tag_pred.bool())
        num_query_tokens = (new_mask == 0).int().sum().item() / batch_size

        ratio = float(num_query_tokens) / num_tokens
        # ic.enable()
        # ic(num_query_tokens, num_tokens, ratio)
        # ic.disable()
        return ratio

    def forward(
        self, src_tokens, src_lengths, 
        prev_output_tokens,
        features_only=False, 
        classification_head_name=None, 
        z_tokens=None, z_lengths=None, 
        z_tags=None, z_tags_prior=None,
        oracle_tag_prob=None,
        tau=None,
        validate=False, 
        tag_only=False,
        **kwargs
    ):
        """
            z_tags: oracle tags
            tag_mode: 
                - '': apply no oracle tags
                - 'pre': apply oracle tags to masking before encoding
                - 'post': apply oracle tags to masking after encoding
        
        """
        if classification_head_name is not None:
            features_only = True

        # we update tau during training, and inject Gumbel noise
        # for validation, we use self.tau, w/o Gumbel noise (tau=None in the batch)
        # inject_noise = False
        # if tau is not None:  # training
        #     self.tau = tau
        #     inject_noise = True

        inject_noise = False if validate else True  # use argmax in val

        if self.tag_on_doc:
            # when tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 640
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            src_tokens, src_lengths = z_tokens, z_lengths
        
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens = True if self.tag_on_doc else False,
            **kwargs,
        )

        if z_tokens is not None:
            assert z_tags is not None, 'z_tags is not a part of inputs'
        
        cached_states, cached_embeddings, cached_mask = None, None, None
        if self.tag_on_doc:
            # states in the last layer is for x
            # and states in the second last layer is for share
            cached_states = encoder_out.encoder_states[-2]  
            cached_embeddings = encoder_out.encoder_embedding
            cached_mask = encoder_out.encoder_padding_mask

        if not self.use_one_encoder:
            ic(z_tags_prior)
            z_encoder_out = self.encoder(
                True,
                z_tokens,
                src_lengths=z_lengths,
                z_tags=z_tags,
                z_tags_prior=z_tags_prior,  # the encoder decides whether to use it or not
                return_all_hiddens=True,
                cached_states=cached_states,
                cached_embeddings=cached_embeddings,
                cached_mask=cached_mask,
                **kwargs,
            )
        else:
            from copy import deepcopy, copy
            z_encoder_out = EncoderOut(
                encoder_out=encoder_out.encoder_out,  # T x B x C
                encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
                encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
                encoder_states=None,  # List[T x B x C]
            )

        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            if self.query_model_lr_multiplier > 1.0:
                tag_scores = GradMultiply.apply(tag_scores, self.query_model_lr_multiplier)

            if tag_only:
                return tag_scores
            
            assert not (self.mask_z_tokens_with_tags and self.mask_z_tokens_with_scores), \
                'You cannot only set one option in (mask_z_tokens_with_tags, mask_z_tokens_with_scores)'
            
            if self.mask_z_tokens_with_tags:
                z_encoder_out = self.update_decoding_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, 
                    oracle_tag_prob=oracle_tag_prob,
                    use_query_tags=False, query_coef=0.0)

            if self.mask_z_tokens_with_scores:
                z_encoder_out = self.soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

            if self.mask_z_tokens_with_gumbel_scores:
                # assert self.tau is not None  # use saved tau in the checkpoint for testing
                z_encoder_out = self.gumbel_soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=tau, inject_noise=inject_noise,
                    use_query_tags=False, query_coef=0.0)
            
            if self.mask_z_tokens_with_straight_through:
                z_encoder_out = self.straight_through(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob)

            if self.mask_z_tokens_with_gumbel_straight_through:
                # assert self.tau is not None
                z_encoder_out = self.gumbel_straight_through(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=tau, inject_noise=inject_noise)

            qt_ratio = self.cal_query_token_ratio(tag_scores, z_encoder_out)


        if self.args.apply_mask_to_shared_enc_layer:
            # the assumption here is the last layer of z_encoder is features for tagging
            # which will not very helpful to summary generation
            # in contrast, the second last layer is shared between the two encoders
            # which contains more semantically-rich token representations
            z_encoder_out = z_encoder_out._replace(encoder_out=z_encoder_out.encoder_states[-2])


        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            z_encoder_out=z_encoder_out,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        
        return x, extra, tag_scores, qt_ratio

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GuidedBARTHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    # logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

    def param_groups(self):
        """
            FIXME find query param names and set different LR for them
            TODO created to be used with customized trainer TwoRLTrainer; currently archived
        """
        query_params = list(filter(lambda p: p.requires_grad and p.dim() == 2,
                                    self.parameters()))
        doc_params = list(filter(lambda p: p.requires_grad and p.dim() == 1,
                                    self.parameters()))
        params = [{'params': query_params}, {'params': doc_params}]
        return params


@register_model('guided_bart_with_tagging_and_std_dec')
class GuidedBARTModelwithTaggingAndStdDec(GuidedTransformerModelwithTaggingAndStdDec):
    """
        Two encoders: shared layers and separate Transformer layers. 
                 Encoding results are concatenated as inputs to the decoder.
        One decoder: standard Transformer decoder
        One tagger on the query encoder.
    """
    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def name(self):
        return 'guided_bart_with_tagging_and_std_dec'

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

        # yumo's revision: add a tagging head
        if args.tagging_head_name:
            self.register_classification_head(name=args.tagging_head_name, num_classes=2, inner_dim=args.encoder_embed_dim)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.mask_z_tokens_with_tags = args.mask_z_tokens_with_tags
        self.mask_z_tokens_with_scores = args.mask_z_tokens_with_scores
        self.mask_score_threshold = args.mask_score_threshold

        logger.info(f'mask_z_tokens_with_tags: {self.mask_z_tokens_with_tags}')
        logger.info(f'mask_z_tokens_with_scores: {self.mask_z_tokens_with_scores}')
        logger.info(f'mask_score_threshold: {self.mask_score_threshold}')

        self.tag_on_doc = args.tag_on_doc
        self.use_one_encoder = args.use_one_encoder
        self.apply_mask_to_shared_enc_layer = args.apply_mask_to_shared_enc_layer
        
    @staticmethod
    def add_args(parser):
        super(GuidedBARTModelwithTaggingAndStdDec, GuidedBARTModelwithTaggingAndStdDec).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )

        parser.add_argument(
            '--tagging-head-name',
            help='head name for tagging'
        )

        parser.add_argument(
            '--mask-z-tokens-with-tags', action="store_true", default=False,
            help='prevent decoding from attending tokens classified as necessary'
        )

        parser.add_argument(
            '--mask-z-tokens-with-scores', action="store_true", default=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--mask-score-threshold', type=float, metavar='D', default=-1.0,
            required=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--fixed-oracle-tag-prob', type=float, metavar='D',
            required=False,
            help='probability of using oracle tags in updating decoder mask, when mask-z-tokens==True'
        )

        parser.add_argument(
            '--warmup-updates-with-oracle-tags', type=int, metavar='D', default=1,
            required=False,
            help='for the first specified steps, use oracle tags for training.'
        )

        parser.add_argument(
            '--oracle-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do oracle probability annealing till this step.'
        )

        parser.add_argument(
            '--oracle-anneal-end-value', type=float, metavar='D',
            required=False,
            help='Do oracle probability annealing till this value.'
        )

        parser.add_argument(
            '--tau-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do tau annealing till this step.'
        )

        parser.add_argument(
            '--tag-on-doc', action="store_true", default=False,
            help='tag on the original document. When set to True, we produced shared representations for x and z, via encoding z.'
        )

        parser.add_argument(
            '--use-one-encoder', action="store_true", default=False,
            help='use only when encoder for x and z. Only allowed when tag-on-doc is used. The last layer in encoder for z is not used.'
        )

        parser.add_argument(
            '--apply-mask-to-shared-enc-layer', action="store_true", default=False,
            help='throw away the last layer of z_encoder and use the second last layer for decoding.'
        )
        

    def get_tags(self, sample):
        """Get tags and their lengths from either the sample"""
        return sample['net_input']['z_tags'], sample['net_input']["z_lengths"]
    
    @property
    def supported_targets(self):
        return {'self'}

    def z_forward_encoder(self,
        src_tokens, src_lengths,
        features_only=False, classification_head_name=None, 
        z_tokens=None, z_lengths=None, z_tags=None, tag_mode='',
        tag_only=None,
        **kwargs
    ):
        """
            For generation.
            z_tags: oracle tags
            tag_mode: 
                - '': oracle tags not applied

        """
        if self.tag_on_doc:
            # we tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 768
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            # logger.info(f'src_tokens: {src_tokens}')
            # logger.info(f'z_tokens: {z_tokens}')
            # logger.info('---------------------------')
            ic(self.tag_on_doc)
            src_tokens, src_lengths = z_tokens, z_lengths
        
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens = True if self.tag_on_doc else False,
            **kwargs,
        )

        # if z_tokens is not None:
            # assert z_tags is not None, 'z_tags is not a part of inputs'

        cached_states, cached_embeddings, cached_mask = None, None, None
        if self.tag_on_doc:
            # states in the last layer is for x
            # and states in the second last layer is for share
            cached_states = encoder_out.encoder_states[-2]  
            cached_embeddings = encoder_out.encoder_embedding
            cached_mask = encoder_out.encoder_padding_mask

        # you can also use z_tags to mask inside self.encoder by setting tag_model
        # but it is just for model analysis
        # at inference, z_tags should only be used for oracle results
        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            z_tags=z_tags,
            tag_mode=tag_mode,
            return_all_hiddens=True,
            cached_states=cached_states,
            cached_embeddings=cached_embeddings,
            cached_mask=cached_mask,
            **kwargs,
        )
        
        oracle_tag_prob = 0.0
        if tag_mode:  # when tag_mode is set, oracle tags are applied
            oracle_tag_prob = 1.0

        ic.enable()
        ic(tag_mode)
        ic(oracle_tag_prob)
        
        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            # ic(tag_scores)
            if tag_only:
                # print('return tag scores')
                return tag_scores
            
            if self.mask_z_tokens_with_tags:
                ic(self.mask_z_tokens_with_tags)
                z_encoder_out = self.update_decoding_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, 
                    oracle_tag_prob=oracle_tag_prob)

            if self.mask_z_tokens_with_scores:
                ic(self.mask_z_tokens_with_scores)
                z_encoder_out = self.soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

        encoder_out = self.integrate_encoders(encoder_out, z_encoder_out)
        ic.disable()
        return encoder_out

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        decoder_out = self.decoder(prev_output_tokens, 
            encoder_out, 
            incremental_state=incremental_state, 
            **extra_args)

        return decoder_out

    def update_decoding_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob):
        """
            tag_scores: B * T * 2

            update encoder_padding_mask with tag_pred back to h_z
            encoder_padding_mask: 1 1 1 1 1 | 0 0 0 0 0
            tag_pred/z_tags: 1 0 1 0 1 | 1 0 1 0 1 (the first 5 should be ignored; attend 7th and 9th tokens)
            results: 1 1 1 1 1 | 1 0 1 0 1 (the first 5 are paddings; attend 7th and 9th tokens))

        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob

        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        
        def _get_estimated():
            tag_pred = argmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            # ic(tag_pred.size())
            # from copy import copy
            # tag_pred_for_analysis = copy(tag_pred)
            # tag_pred_for_analysis[z_encoder_out.encoder_padding_mask==True] = 2
            # num_zeros = (tag_pred_for_analysis == 0).int().sum().item() / float(tag_pred.size(0))
            # num_ones = (tag_pred_for_analysis == 1).int().sum().item() / float(tag_pred.size(0))
            # zero_ratio = num_zeros/(num_zeros+num_ones)
            # ic(f'keywords: {num_zeros}, masks: {num_ones}, ratio of keywords: {zero_ratio}')
            return logical_or(z_encoder_out.encoder_padding_mask, tag_pred.bool())

        def _get_oracle():
            return logical_or(z_encoder_out.encoder_padding_mask, z_tags.bool())

        if oracle_tag_prob == 1.0:
            updated_mask = _get_oracle()
        elif oracle_tag_prob == 0.0:
            updated_mask = _get_estimated()
        else:
            odd = np.random.rand()
            updated_mask = _get_oracle() if odd <= oracle_tag_prob else _get_estimated()

        def _fix_all_mask_case(updated_mask):
            """
                it is possible that all estimations are zero, which will cause NaN problems.
                in this case, we just use the original mask, which masks nothing but pads.
            """
            all_mask_indices = torch.sum(updated_mask, dim=1) == updated_mask.size(1)

            if all_mask_indices.any():
                n_indices = all_mask_indices.sum()
                logger.info(f'All tokens are masked in {n_indices} samples. Keep the original mask.')
                updated_mask[all_mask_indices] = z_encoder_out.encoder_padding_mask[all_mask_indices]
            
            return updated_mask

        updated_mask = _fix_all_mask_case(updated_mask)
        return z_encoder_out._replace(encoder_padding_mask=updated_mask)
        
    def soft_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob, threshold):
        """
            tag_scores: B * T * 2
            z_tags: B * T
        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob
        
        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        
        odd = np.random.rand()
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            soft_mask = (1.0 - z_tags).half()
            soft_mask = torch.unsqueeze(soft_mask, dim=-1)
        
        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            tag_dist = F.softmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            soft_mask = tag_dist[:, :, 0:1]
            if threshold > 0:
                soft_mask = (soft_mask>=threshold).float().half()

                # n_ones = torch.sum(torch.squeeze(soft_mask, dim=-1), dim=-1)
                # ic.enable()
                # ic(threshold)
                # ic(n_ones)
                # ic.disable()

        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = soft_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        # ic(z_encoder_out.encoder_out.size())
        # ic(soft_mask.size())
        # ic(updated_encoder_out.size())
        
        return z_encoder_out._replace(encoder_out=updated_encoder_out)
        
    def integrate_encoders(self, enc_1, enc_2):
        encoder_out = torch.cat([enc_1.encoder_out, enc_2.encoder_out], 0)  # T x B x C
        encoder_padding_mask = torch.cat([enc_1.encoder_padding_mask, enc_2.encoder_padding_mask], 1) # B x T
        encoder_embedding = torch.cat([enc_1.encoder_embedding, enc_2.encoder_embedding], 1)  # B x T x C
        
        return EncoderOut(
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=None,  # List[T x B x C]
        )
    
    def forward(
        self, src_tokens, src_lengths, 
        prev_output_tokens,
        features_only=False, 
        classification_head_name=None, 
        z_tokens=None, z_lengths=None, z_tags=None,
        oracle_tag_prob=None,
        tag_only=False,
        **kwargs
    ):
        """
            z_tags: oracle tags
            tag_mode: 
                - '': apply no oracle tags
                - 'pre': apply oracle tags to masking before encoding
                - 'post': apply oracle tags to masking after encoding
        
        """
        if classification_head_name is not None:
            features_only = True

        if self.tag_on_doc:
            # we tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 768
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            # logger.info(f'src_tokens: {src_tokens}')
            # logger.info(f'z_tokens: {z_tokens}')
            # logger.info('---------------------------')
            src_tokens, src_lengths = z_tokens, z_lengths
        
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens = True if self.tag_on_doc else False,
            **kwargs,
        )

        if z_tokens is not None:
            assert z_tags is not None, 'z_tags is not a part of inputs'
        
        cached_states, cached_embeddings, cached_mask = None, None, None
        if self.tag_on_doc:
            # states in the last layer is for x
            # and states in the second last layer is for share
            cached_states = encoder_out.encoder_states[-2]  
            cached_embeddings = encoder_out.encoder_embedding
            cached_mask = encoder_out.encoder_padding_mask

        if not self.use_one_encoder:
            z_encoder_out = self.encoder(
                True,
                z_tokens,
                src_lengths=z_lengths,
                z_tags=z_tags,
                return_all_hiddens=True,
                cached_states=cached_states,
                cached_embeddings=cached_embeddings,
                cached_mask=cached_mask,
                **kwargs,
            )
        else:
            from copy import deepcopy, copy
            z_encoder_out = EncoderOut(
                encoder_out=encoder_out.encoder_out,  # T x B x C
                encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
                encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
                encoder_states=None,  # List[T x B x C]
            )

        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            if tag_only:
                return tag_scores
            
            assert not (self.mask_z_tokens_with_tags and self.mask_z_tokens_with_scores), \
                'You cannot only set one option in (mask_z_tokens_with_tags, mask_z_tokens_with_scores)'
            
            if self.mask_z_tokens_with_tags:
                z_encoder_out = self.update_decoding_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, 
                    oracle_tag_prob=oracle_tag_prob)

            if self.mask_z_tokens_with_scores:
                z_encoder_out = self.soft_mask(z_encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

        if self.args.apply_mask_to_shared_enc_layer:
            # the assumption here is the last layer of z_encoder is features for tagging
            # which will not very helpful to summary generation
            # in contrast, the second last layer is shared between the two encoders
            # which contains more semantically-rich token representations
            z_encoder_out = z_encoder_out._replace(encoder_out=z_encoder_out.encoder_states[-2])
        
        encoder_out = self.integrate_encoders(encoder_out, z_encoder_out)
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra, tag_scores

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GuidedBARTHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    # logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


@register_model('guided_bart_with_tagging_and_std_enc_dec')
class GuidedBARTModelwithTaggingAndStdEncDec(GuidedTransformerModelwithTaggingAndStdEncDec):
    """
        One encoder: standard Transformer encoder
        One decoder: standard Transformer decoder
        One tagger on the encoder
    """

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def name(self):
        return 'guided_bart_with_tagging_and_std_enc_dec'

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

        # yumo's revision: add a tagging head
        if args.tagging_head_name:
            self.register_classification_head(name=args.tagging_head_name, num_classes=2, inner_dim=args.encoder_embed_dim)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.mask_z_tokens_with_tags = args.mask_z_tokens_with_tags
        self.mask_z_tokens_with_scores = args.mask_z_tokens_with_scores
        self.mask_z_tokens_with_gumbel_scores = args.mask_z_tokens_with_gumbel_scores

        self.mask_score_threshold = args.mask_score_threshold
        self.mask_z_tokens_with_straight_through = args.mask_z_tokens_with_straight_through
        self.mask_z_tokens_with_gumbel_straight_through = args.mask_z_tokens_with_gumbel_straight_through

        if self.mask_z_tokens_with_tags:
            logger.info(f'mask_z_tokens_with_tags: {self.mask_z_tokens_with_tags}')

        if self.mask_z_tokens_with_scores:
            logger.info(f'mask_z_tokens_with_scores: {self.mask_z_tokens_with_scores}')
            logger.info(f'mask_score_threshold: {self.mask_score_threshold}')

        if self.mask_z_tokens_with_straight_through:
            logger.info(f'mask_z_tokens_with_straight_through: {self.mask_z_tokens_with_straight_through}')

        if self.mask_z_tokens_with_gumbel_straight_through:
            logger.info(f'mask_z_tokens_with_gumbel_straight_through: {self.mask_z_tokens_with_gumbel_straight_through}')

        if self.mask_z_tokens_with_gumbel_scores:
            logger.info(f'mask_z_tokens_with_gumbel_scores: {self.mask_z_tokens_with_gumbel_scores}')

        self.tag_on_doc = args.tag_on_doc
        self.use_one_encoder = args.use_one_encoder
        self.apply_mask_to_shared_enc_layer = args.apply_mask_to_shared_enc_layer

        self.query_model_lr_multiplier = args.query_model_lr_multiplier

        # tau is an attribute in args which is dynamically set during training
        # so self.tau == None during training
        # during training/validation, we feed tau via batch (see train.py), and also update it in args
        # during tesing, we have arg.tau and use it as self.tau
        self.tau = args.tau if hasattr(args, 'tau') else None
        logger.info(f'Set self.tau to: {self.tau}')
        
    @staticmethod
    def add_args(parser):
        super(GuidedBARTModelwithTaggingAndStdEncDec, GuidedBARTModelwithTaggingAndStdEncDec).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )

        parser.add_argument(
            '--tagging-head-name',
            help='head name for tagging'
        )

        parser.add_argument(
            '--mask-z-tokens-with-tags', action="store_true", default=False,
            help='prevent decoding from attending tokens classified as necessary'
        )

        parser.add_argument(
            '--mask-z-tokens-with-scores', action="store_true", default=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--mask-z-tokens-with-gumbel-scores', action="store_true", default=False,
            help='weigh representations with their salience scores injected with gumbel noise'
        )

        parser.add_argument(
            '--mask-z-tokens-with-straight-through', action="store_true", default=False,
            help='hard and differentiable masking with straight through trick'
        )

        parser.add_argument(
            '--mask-z-tokens-with-gumbel-straight-through', action="store_true", default=False,
            help='hard and differentiable masking with sampling and straight through trick'
        )
        
        parser.add_argument(
            '--mask-score-threshold', type=float, metavar='D', default=-1.0,
            required=False,
            help='weigh representations with their salience scores'
        )

        parser.add_argument(
            '--query-model-lr-multiplier', type=float, metavar='D', default=1.0,
            required=False,
            help='increase the learning rate by this multiplier for params in the query model'
        )

        parser.add_argument(
            '--fixed-oracle-tag-prob', type=float, metavar='D',
            required=False,
            help='probability of using oracle tags in updating decoder mask, when mask-z-tokens==True'
        )

        parser.add_argument(
            '--warmup-updates-with-oracle-tags', type=int, metavar='D', default=1,
            required=False,
            help='for the first specified steps, use oracle tags for training.'
        )

        parser.add_argument(
            '--oracle-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do oracle probability annealing till this step.'
        )

        parser.add_argument(
            '--oracle-anneal-end-value', type=float, metavar='D',
            required=False,
            help='Do oracle probability annealing till this value.'
        )

        parser.add_argument(
            '--tau-anneal-end-step', type=int, metavar='D',
            required=False,
            help='Do tau annealing till this step.'
        )

        parser.add_argument(
            '--tag-on-doc', action="store_true", default=False,
            help='tag on the original document. When set to True, we produced shared representations for x and z, via encoding z.'
        )

        parser.add_argument(
            '--use-one-encoder', action="store_true", default=False,
            help='use only when encoder for x and z. Only allowed when tag-on-doc is used. The last layer in encoder for z is not used.'
        )

        parser.add_argument(
            '--apply-mask-to-shared-enc-layer', action="store_true", default=False,
            help='throw away the last layer of z_encoder and use the second last layer for decoding.'
        )

        parser.add_argument(
            '--fixed-tau', type=float, metavar='D',
            required=False,
            help='Constant tau for temperature controlling in Gumbel Softmax.'
        )

        parser.add_argument(
            '--tau', type=float, metavar='D',
            required=False,
            help='Dynamic tau for temperature controlling in Gumbel Softmax.'
        )

        parser.add_argument(
            '--use-prior', action="store_true", default=False,
            help='use prior query information, sampled or given.'
        )

        parser.add_argument(
            '--max-num-prior-spans', type=int, metavar='D',
            default=None,
            required=False,
            help='the max number of prior spans allowed to be sampled for query prior.'
        )

        parser.add_argument(
            '--max-prior-ratio', type=float, metavar='D',
            default=None,
            required=False,
            help='the max number of tokens allowed for query prior, as a ratio of total number of query tokens.'
        )
        
    def get_tags(self, sample):
        """Get tags and their lengths from either the sample"""
        return sample['net_input']['z_tags'], sample['net_input']["z_lengths"]
    
    @property
    def supported_targets(self):
        return {'self'}

    def z_forward_encoder(self,
        src_tokens, src_lengths,
        features_only=False, classification_head_name=None, 
        z_tokens=None, z_lengths=None, z_tags=None, tag_mode='',
        tag_only=None,
        inject_noise=False,
        **kwargs
    ):
        """
            For generation.
            z_tags: oracle tags
            tag_mode: 
                - '': oracle tags not applied

        """
        if self.tag_on_doc:
            # we tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 768
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            # logger.info(f'src_tokens: {src_tokens}')
            # logger.info(f'z_tokens: {z_tokens}')
            # logger.info('---------------------------')
            ic(self.tag_on_doc)
            src_tokens, src_lengths = z_tokens, z_lengths
        
        del kwargs['z_tags_prior']

        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )

        def _use_query_tags(tag_mode, default_query_coef=0.0):
            """
                tag_mode: query_1.0
            """
            if 'query' in tag_mode or 'q' in tag_mode:
                tag_items = tag_mode.split('_')
                if len(tag_items) >= 2:
                    query_coef = float(tag_items[-1])
                    return True, query_coef
                return True, default_query_coef
            
            return False, default_query_coef

        ic.enable()
        
        use_query_tags, query_coef = _use_query_tags(tag_mode, default_query_coef=0.0)
        ic(use_query_tags)
        ic(query_coef)
        
        oracle_tag_prob = 0.0
        if tag_mode:  # when tag_mode is set, oracle tags are applied
            oracle_tag_prob = 1.0

        ic.enable()
        ic(tag_mode)
        ic(oracle_tag_prob)
        
        if self.args.tagging_head_name in self.classification_heads:
            seq_len, batch_size, hidden_dim = encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            # ic(tag_scores)
            if tag_only:
                # print('return tag scores')
                return tag_scores
            
            if self.mask_z_tokens_with_tags:
                ic(self.mask_z_tokens_with_tags)
                encoder_out = self.update_decoding_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, 
                    oracle_tag_prob=oracle_tag_prob)

            if self.mask_z_tokens_with_scores:
                ic(self.mask_z_tokens_with_scores)
                encoder_out = self.soft_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

            if self.mask_z_tokens_with_gumbel_scores:
                ic(self.tau)
                assert self.tau is not None  # use saved tau in the checkpoint for testing
                encoder_out = self.gumbel_soft_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=self.tau, inject_noise=inject_noise,
                    use_query_tags=use_query_tags, query_coef=query_coef)

        ic.disable()
        return encoder_out

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        decoder_out = self.decoder(prev_output_tokens, 
            encoder_out, 
            incremental_state=incremental_state, 
            **extra_args)

        return decoder_out

    def update_decoding_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob):
        """
            tag_scores: B * T * 2

            update encoder_padding_mask with tag_pred back to h_z
            encoder_padding_mask: 1 1 1 1 1 | 0 0 0 0 0
            tag_pred/z_tags: 1 0 1 0 1 | 1 0 1 0 1 (the first 5 should be ignored; attend 7th and 9th tokens)
            results: 1 1 1 1 1 | 1 0 1 0 1 (the first 5 are paddings; attend 7th and 9th tokens))

        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob

        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        
        def _get_estimated():
            tag_pred = argmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            # ic(tag_pred.size())
            # from copy import copy
            # tag_pred_for_analysis = copy(tag_pred)
            # tag_pred_for_analysis[z_encoder_out.encoder_padding_mask==True] = 2
            # num_zeros = (tag_pred_for_analysis == 0).int().sum().item() / float(tag_pred.size(0))
            # num_ones = (tag_pred_for_analysis == 1).int().sum().item() / float(tag_pred.size(0))
            # zero_ratio = num_zeros/(num_zeros+num_ones)
            # ic(f'keywords: {num_zeros}, masks: {num_ones}, ratio of keywords: {zero_ratio}')
            return logical_or(z_encoder_out.encoder_padding_mask, tag_pred.bool())

        def _get_oracle():
            return logical_or(z_encoder_out.encoder_padding_mask, z_tags.bool())

        if oracle_tag_prob == 1.0:
            updated_mask = _get_oracle()
        elif oracle_tag_prob == 0.0:
            updated_mask = _get_estimated()
        else:
            odd = np.random.rand()
            updated_mask = _get_oracle() if odd <= oracle_tag_prob else _get_estimated()

        def _fix_all_mask_case(updated_mask):
            """
                it is possible that all estimations are zero, which will cause NaN problems.
                in this case, we just use the original mask, which masks nothing but pads.
            """
            all_mask_indices = torch.sum(updated_mask, dim=1) == updated_mask.size(1)

            if all_mask_indices.any():
                n_indices = all_mask_indices.sum()
                logger.info(f'All tokens are masked in {n_indices} samples. Keep the original mask.')
                updated_mask[all_mask_indices] = z_encoder_out.encoder_padding_mask[all_mask_indices]
            
            return updated_mask

        updated_mask = _fix_all_mask_case(updated_mask)
        return z_encoder_out._replace(encoder_padding_mask=updated_mask)
    
    def soft_mask(self, z_encoder_out, tag_scores, z_tags, 
            oracle_tag_prob, threshold):
        """
            tag_scores: B * T * 2
            z_tags: B * T
        """

        # during training, override oracle_tag_prob with fixed prob
        # this does not apply to inference
        # if you want to use oracle tags at inference
        # you can set tag_mode in z_forward_encoder, which then sets oracle_tag_prob to 1.0
        # if not inference and self.fixed_oracle_tag_prob >= 0.0:  
            # oracle_tag_prob = self.fixed_oracle_tag_prob
        
        # assert oracle_tag_prob is not None
        
        # if inference:
            # ic(oracle_tag_prob)
        
        odd = np.random.rand()
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            soft_mask = (1.0 - z_tags).half()
            soft_mask = torch.unsqueeze(soft_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            tag_dist = F.softmax(tag_scores, dim=-1)  # B * T, 0: to reserve, 1: to mask
            soft_mask = tag_dist[:, :, 0:1]
            if threshold > 0:
                soft_mask = (soft_mask>=threshold).float().half()

        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = soft_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        return z_encoder_out._replace(encoder_out=updated_encoder_out)
    
    def gumbel_soft_mask(self, z_encoder_out, tag_scores, z_tags, oracle_tag_prob, tau, inject_noise, 
            use_query_tags, query_coef):
        ic.enable()
        assert tau, 'specify tau when in gumbel_softmax!'

        odd = np.random.rand()
        if oracle_tag_prob == 1.0 or odd <= oracle_tag_prob:
            soft_mask = (1.0 - z_tags).half()
            soft_mask = torch.unsqueeze(soft_mask, dim=-1)

        if oracle_tag_prob == 0.0 or odd > oracle_tag_prob:
            tag_dist = gumbel_softmax(logits=tag_scores, tau=tau, hard=False, inject_noise=inject_noise)
            soft_mask = tag_dist[:, :, 0:1]  # B * T * 1

        def _inject_query_prior(soft_mask, z_tags, query_coef, padding_mask):
            """
                Incorporate prior query information from `z_tags` to `soft_mask`, 
                controlled via query_coef.

                soft_mask: B * T * 1
                z_tags: B * T
                query_coef: float
                    - [0.0, 1.0]: linear interpolation
                    - 10: set to max
                    - 11: set to 1.0
                padding_mask: B * T

                return: updated soft_mask
            """
            # ic(query_tags.size())
            # ic(soft_mask.size())
            if 0.0 <= query_coef <= 1.0:  # linear interpolation
                query_tags = (1.0 - z_tags).half()
                query_tags = torch.unsqueeze(query_tags, dim=-1)
                soft_mask = query_coef * query_tags + (1-query_coef) * soft_mask
                return soft_mask
            
            if query_coef == 10:  # set to max
                query_tags = (1.0 - z_tags).bool()
                # ic(padding_mask.size())
                soft_mask = soft_mask.squeeze(-1)  # B * T
                soft_mask.masked_fill_(padding_mask, 0.)  # B * T
                
                max_values = torch.max(soft_mask, dim=-1, keepdim=False)[0]  # B,
                # ic(max_values)
                seq_len = padding_mask.size(-1)
                max_values = torch.stack([max_values]*seq_len, dim=-1)  # B * T
                # ic(max_values.size())

                soft_mask = torch.where(query_tags, max_values, soft_mask)  # set query tokens to max_values; B * T
                soft_mask = soft_mask.unsqueeze(-1)  # B * T * 1
                return soft_mask
            
            if query_coef == 11:  # set to 1.0
                query_tags = (1.0 - z_tags).bool()
                soft_mask = soft_mask.squeeze(-1)  # B * T
                soft_mask = torch.where(query_tags, query_tags.half(), soft_mask)  # set query tokens to 1.0; B * T
                soft_mask = soft_mask.unsqueeze(-1)  # B * T * 1
                return soft_mask
        
        if use_query_tags:
            soft_mask = _inject_query_prior(soft_mask, z_tags, query_coef, 
                padding_mask=z_encoder_out.encoder_padding_mask)
        
        # z_encoder_out.encoder_out: T * B * 1024
        updated_encoder_out = soft_mask.transpose(0, 1) * z_encoder_out.encoder_out
        
        ic.disable()
        return z_encoder_out._replace(encoder_out=updated_encoder_out)

    def forward(
        self, src_tokens, src_lengths, 
        prev_output_tokens,
        features_only=False, 
        classification_head_name=None, 
        z_tokens=None, z_lengths=None, 
        z_tags=None, z_tags_prior=None,
        oracle_tag_prob=None,
        tau=None,
        validate=False, 
        tag_only=False,
        **kwargs
    ):
        """
            z_tags: oracle tags
            tag_mode: 
                - '': apply no oracle tags
                - 'pre': apply oracle tags to masking before encoding
                - 'post': apply oracle tags to masking after encoding
        
        """
        if classification_head_name is not None:
            features_only = True

        inject_noise = False if validate else True  # use argmax in val

        if self.tag_on_doc:
            # we tag_on_doc is set True, we set MAX_Z_TOKENS to, at least, 768
            # and z is treated as x
            # we do not do the other way around (treating x as z)
            # because the tags are produced as per z
            # and there could be minor differences between z and x
            # logger.info(f'src_tokens: {src_tokens}')
            # logger.info(f'z_tokens: {z_tokens}')
            # logger.info('---------------------------')
            src_tokens, src_lengths = z_tokens, z_lengths
        
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )

        if z_tokens is not None:
            assert z_tags is not None, 'z_tags is not a part of inputs'

        if self.args.tagging_head_name in self.classification_heads:
            seq_len, batch_size, hidden_dim = encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(seq_len, batch_size, -1).transpose(0, 1)  # B * T * 1

            if tag_only:
                return tag_scores
            
            assert not (self.mask_z_tokens_with_tags and self.mask_z_tokens_with_scores), \
                'You cannot only set one option in (mask_z_tokens_with_tags, mask_z_tokens_with_scores)'

            if self.mask_z_tokens_with_tags:
                encoder_out = self.update_decoding_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, 
                    oracle_tag_prob=oracle_tag_prob,
                    use_query_tags=False, query_coef=0.0)

            if self.mask_z_tokens_with_scores:
                encoder_out = self.soft_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags,
                    oracle_tag_prob=oracle_tag_prob,
                    threshold=self.mask_score_threshold)

            if self.mask_z_tokens_with_gumbel_scores:
                encoder_out = self.gumbel_soft_mask(encoder_out, tag_scores=tag_scores, 
                    z_tags=z_tags, oracle_tag_prob=oracle_tag_prob, tau=tau, inject_noise=inject_noise,
                    use_query_tags=False, query_coef=0.0)
        
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra, tag_scores

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GuidedBARTHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_tagging_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTTaggingHead(
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue
            
            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]

            print(f'{k}: {head_name}')
            if head_name == 'sigmoid_bce':
                continue

            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    # logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BARTTaggingHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim,
        num_classes,
        pooler_dropout,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        return x


@register_model_architecture('guided_bart_with_tagging', 'guided_bart_large_with_tagging')
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    # yumo's revision: set tagging head (default: binary classification)
    args.tagging_head_name = getattr(args, 'tagging_head_name', 'binary_classification')


@register_model_architecture("guided_bart_with_tagging", "guided_bart_base_with_tagging")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)


@register_model_architecture("guided_bart_with_tagging_and_std_dec", "guided_bart_large_with_tagging_and_std_dec")
def bart_large_std_dec_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    # yumo's revision: set tagging head (default: binary classification)
    args.tagging_head_name = getattr(args, 'tagging_head_name', 'binary_classification')


@register_model_architecture("guided_bart_with_tagging_and_std_dec", "guided_bart_base_with_tagging_and_std_dec")
def bart_base_std_dec_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)


@register_model_architecture("guided_bart_with_tagging_and_std_enc_dec", "guided_bart_large_with_tagging_and_std_enc_dec")
def bart_large_std_enc_dec_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    # yumo's revision: set tagging head (default: binary classification)
    args.tagging_head_name = getattr(args, 'tagging_head_name', 'binary_classification')


@register_model_architecture("guided_bart_with_tagging_and_std_enc_dec", "guided_bart_base_with_tagging_and_std_enc_dec")
def bart_base_std_enc_dec_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
