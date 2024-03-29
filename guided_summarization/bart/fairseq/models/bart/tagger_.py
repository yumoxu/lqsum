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
from fairseq.models.guided_transformer import GuidedTransformerModelwithTagging
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_encoder import EncoderOut

from .guided_hub_interface import GuidedBARTHubInterface
#from .hub_interface import BARTHubInterface
from fairseq.modules import (
    AdaptiveSoftmax,
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
        return 'tagger'

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
        **kwargs
    ):
        """
            For generation.
            z_tags: oracle tags
            tag_mode: 
                - '': oracle tags not applied

        """
        # you can also use z_tags to mask inside self.encoder by setting tag_model
        # but it is just for model analysis
        # at inference, z_tags should only be used for oracle results
        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            z_tags=z_tags,
            tag_mode=tag_mode,
            **kwargs,
        )
        
        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

        return z_encoder_out        

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
        if z_tokens is not None:
            assert z_tags is not None, 'z_tags is not a part of inputs'

        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            z_tags=z_tags,
            return_all_hiddens=True,
            **kwargs,
        )

        if self.args.tagging_head_name in self.classification_heads:
            z_seq_len, batch_size, hidden_dim = z_encoder_out.encoder_out.size()
            tag_scores = self.classification_heads[self.args.tagging_head_name](
                z_encoder_out.encoder_out.view(-1, hidden_dim)
            ).view(z_seq_len, batch_size, -1).transpose(0, 1)  # B * T * 2

            return tag_scores

        return z_encoder_out

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
