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

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.guided_transformer import GuidedTransformerModel
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
)
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model('guided_bart')
class GuidedBARTModel(GuidedTransformerModel):

    @classmethod
    def hub_models(cls):
        return {
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def name(self):
        return 'guided_bart'

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization

        #self.f1 = GuidedTransformerEncoderLayer(args)
        #self.f2 = GuidedTransformerEncoderLayer(args)
        #self.f1 = TransformerEncoderLayer(args)
        #self.f2 = TransformerEncoderLayer(args)
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    @staticmethod
    def add_args(parser):
        super(GuidedBARTModel, GuidedBARTModel).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )

    @property
    def supported_targets(self):
        return {'self'}

    def z_forward_encoder(self,
        src_tokens, src_lengths,
        features_only=False, classification_head_name=None, 
        z_tokens=None, z_lengths=None, **kwargs
    ):
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            **kwargs,
        )
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
        #attn_args = {
        #    "alignment_layer": self.alignment_layer,
        #    "alignment_heads": self.alignment_heads,
        #}
        decoder_out = self.decoder(prev_output_tokens, encoder_out, z_encoder_out=z_encoder_out, incremental_state=incremental_state, **extra_args)

        return decoder_out

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, 
        z_tokens=None, z_lengths=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        #FIXME: for pretrain
        #z_src_tokens = src_tokens.clone()
        #z_src_lengths = src_lengths

        #encoder_out, encoder_padding_mask, encoder_embedding, encoder_states = self.encoder(
        encoder_out = self.encoder(
            False,
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        #encoder_out.encoder_out = self.f1(encoder_out.encoder_out, encoder_out.encoder_padding_mask)
        #print(src_tokens.size())
        #print(z_tokens.size())
        #print(src_tokens)
        #print(z_tokens)
        #print(z_tokens)
        #encoder_out = self.f1(encoder_out, encoder_padding_mask)
        #encoder_out = encoder_out + encoder_out2
        '''x = self.f1(encoder_out.encoder_out, encoder_out.encoder_padding_mask)'''
        #encoder_out = EncoderOut(
        #    encoder_out=encoder_out,  # T x B x C
        #    encoder_padding_mask=encoder_padding_mask,  # B x T
        #    encoder_embedding=encoder_embedding,  # B x T x C
        #    encoder_states=encoder_states,  # List[T x B x C]
        #)

        #z_encoder_out, z_encoder_padding_mask, z_encoder_embedding, z_encoder_states = self.encoder(
        z_encoder_out = self.encoder(
            True,
            z_tokens,
            src_lengths=z_lengths,
            **kwargs,
        )
        #z_encoder_out.encoder_out = self.f2(z_encoder_out.encoder_out, encoder_out.encoder_padding_mask)
        #x = self.f2(z_encoder_out.encoder_out, z_encoder_out.encoder_padding_mask)
        #z_encoder_out = self.f2(z_encoder_out, z_encoder_padding_mask)
        #z_encoder_out = z_encoder_out + z_encoder_out2
        #z_encoder_out = EncoderOut(
        #    encoder_out=z_encoder_out,  # T x B x C
        #    encoder_padding_mask=z_encoder_padding_mask,  # B x T
        #    encoder_embedding=z_encoder_embedding,  # B x T x C
        #    encoder_states=z_encoder_states,  # List[T x B x C]
        #)

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
        return x, extra

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
                    logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v
