#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger('fairseq_cli.train')


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def my_entropy(lprobs, pad_mask, reduce=True):
    """
        entropy = -\sum p * log p
        The input tensor `lprob` is log p.
        Therefore, p = exp(lprobs), and p * log p = exp(lprobs) * lprobs.
    """
    # entropy = -torch.where(~pad_mask, lprobs * lprobs.log(), lprobs.new([0.0])).sum(dim=-1, keepdim=False)  # (B*T)
    entropy = -torch.where(~pad_mask, torch.exp(lprobs) * lprobs, lprobs.new([0.0])).sum(dim=-1, keepdim=False)  # (B*T)
    if reduce:
        entropy = entropy.sum()
    return entropy


def entropy(lprobs, pad_mask, reduce=True):
    probs = F.softmax(lprobs, -1) + 1e-8
    entropy = -probs * torch.log(probs)
    entropy = torch.sum(entropy, -1, keepdim=True)
    # print(f'entropy: {entropy.size()}')
    # print(f'pad_mask: {pad_mask.size()}')
    if pad_mask.any():
        entropy.masked_fill_(pad_mask, 0.)
    
    if reduce:
        entropy = entropy.sum()
    return entropy


@register_criterion('guided_label_smoothed_cross_entropy')
class GuidedLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('guided_label_smoothed_cross_entropy_for_tagging')
class GuidedLabelSmoothedCrossEntropyCriterionforTagging(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.tag_label_smoothing
        self.padding_idx = 2

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--tag-label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, tag_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(tag_output, log_probs=True)
        # print(f'lprobs: {lprobs.size()}')
        lprobs = lprobs.view(-1, lprobs.size(-1))
        tags, _ = model.get_tags(sample)
        tags = tags.view(-1, 1)
        # print(f'tags: {tags.size()}')

        # print(f'[compute_loss] tags: {tags}')
        # print(f'[compute_loss] lprobs: {lprobs}')

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, tags, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, lprobs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('guided_posterior_entropy_loss')
class GuidedPosteriorEntropyLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.padding_idx = 2

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, lprobs, sample, reduce=True):
        """
            lprobs: -1 * 2
        """
        tags, _ = model.get_tags(sample)
        tags = tags.view(-1, 1)
        pad_mask = tags.eq(self.padding_idx)
        # pad_mask = torch.stack([pad_mask, pad_mask], dim=-1)  # -1, 2
        # in ELBO, we maximize the entropy. Thus, we minimize -1 * entropy.
        loss = -entropy(lprobs, pad_mask, reduce=reduce)  
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('guided_label_smoothed_bce_for_tagging')
class GuidedLabelSmoothedBCEforTagging(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.tag_label_smoothing
        self.padding_idx = 2

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--tag-label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, tag_output, sample, reduce=True):
        mask_prob = tag_output[0]  # B * T
        # ic(mask_prob)
        query_prob = 1.0 - mask_prob
        
        probs = torch.cat([query_prob, mask_prob], dim=-1)
        lprobs = torch.log(probs)
        # ic(lprobs.size())

        lprobs = lprobs.view(-1, lprobs.size(-1))
        tags, _ = model.get_tags(sample)
        tags = tags.view(-1, 1)
        # print(f'tags: {tags.size()}')
        
        # print(f'[compute_loss] tags: {tags}')
        # print(f'[compute_loss] lprobs: {lprobs}')

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, tags, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('tagging_generation_joint_criterion')
class TaggingGenerationJointCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.lm_criterion = GuidedLabelSmoothedCrossEntropyCriterion(args, task)
        self.tag_criterion = GuidedLabelSmoothedCrossEntropyCriterionforTagging(args, task)
        self.post_entropy_criterion = GuidedPosteriorEntropyLoss(args, task)

        self.tag_coef = args.tag_coef
        self.post_entropy_coef = args.post_entropy_coef if hasattr(args, 'post_entropy_coef') else 0.0

        self.tag_only = args.tag_only
        self.norm_loss_in_sample = args.norm_loss_in_sample

        self.decrease_tag_coef_over_steps = args.decrease_tag_coef_over_steps

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--tag-coef', default=1.0, type=float,
                            help='coefficient to combine tagging loss. 0 means no tagging loss.')
        parser.add_argument('--post-entropy-coef', default=0.0, type=float,
                            help='coefficient to combine posterior entropy loss. 0 means no posterior entropy loss.')

        parser.add_argument('--decrease-tag-coef-over-steps', default=-1, type=int,
                            help='decrease self.tag-coef over this specified number of trainig steps \
                                from its original value to 0.0.')
        
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing in generation, 0 means no label smoothing')
        parser.add_argument('--tag-label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing in tagging, 0 means no label smoothing')
        
        parser.add_argument('--norm-loss-in-sample', action='store_true', default=False,
                            help='get token-level loss and set sample_size to 1.')
        parser.add_argument('--tag-only', action='store_true', default=False,
                            help='train only a tagger using only tagging loss')

    def get_scheduled_tag_coef(self, num_updates):
        """
            decrease tag coeffient from self.tag_coef till 0.0 over steps defined by self.decrease_tag_coef_over_steps.
        """
        current_step = num_updates
        max_step = self.decrease_tag_coef_over_steps
        start, end = self.tag_coef, 0.01
        
        step = (start-end) / max_step
        return max(start - step * current_step, end)
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], tag_only=self.tag_only)
        loss, lm_loss, tag_loss, entropy_loss, qt_ratio = self.compute_loss(model, net_output, sample, reduce=reduce)

        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        assert not self.args.sentence_avg
        
        if self.norm_loss_in_sample:
            sample_size = 1
        else:
            sample_size = sample['ntokens'] + sample['n_z_tokens']
        
        logging_output = {
            'loss': loss.data,
            'lm_loss': lm_loss.data if lm_loss is not None else 0.0,
            'tag_loss': tag_loss.data if tag_loss is not None else 0.0,
            'entropy_loss': entropy_loss.data if entropy_loss is not None else 0.0,
            # 'nll_loss': lm_nll_loss.data,
            'ntokens': sample['ntokens'],
            'n_z_tokens': sample['n_z_tokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'norm_loss_in_sample': self.norm_loss_in_sample,
            'qt_ratio': qt_ratio if qt_ratio is not None else 0.0,
            # 'tag_coef': tag_coef,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # assert len(net_output)==3, f'net_output has {len(net_output)} elements'
        qt_ratio = None
        if self.tag_only:
            if len(net_output) == 1:
                tag_scores = net_output
            else:
                tag_scores, qt_ratio = net_output
        else:
            if len(net_output) == 3:
                x, extra, tag_scores = net_output
            else:
                x, extra, tag_scores, qt_ratio = net_output
        
        if self.decrease_tag_coef_over_steps > 1:
            num_updates = sample['num_updates']
            tag_coef = self.get_scheduled_tag_coef(num_updates)
            if num_updates % 100 == 0:
                logger.info(f'Training steps: {num_updates}, tag_coef: {tag_coef}')
        else:
            tag_coef = self.tag_coef
            
        lm_loss, tag_loss, entropy_loss = None, None, None
        if tag_coef >= 0.0:
            tag_loss, _, lprobs = self.tag_criterion.compute_loss(model, tag_output=[tag_scores,], sample=sample, reduce=reduce)
            if self.norm_loss_in_sample:
                tag_loss = tag_loss / sample['n_z_tokens']
            
            # TODO test entropy_loss
            if self.post_entropy_coef > 0.0:
                entropy_loss = self.post_entropy_criterion.compute_loss(model, lprobs, sample, reduce=True)
                if self.norm_loss_in_sample:
                    entropy_loss = entropy_loss / sample['n_z_tokens']

        if not self.tag_only:
            lm_loss, _ = self.lm_criterion.compute_loss(model, net_output=[x, extra], sample=sample, reduce=reduce)
            if self.norm_loss_in_sample:
                lm_loss = lm_loss / sample['ntokens']
        
        if lm_loss is not None and tag_loss is not None:
            loss = lm_loss + tag_coef * tag_loss
            if entropy_loss is not None:
                loss = loss + self.post_entropy_coef * entropy_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif tag_loss is not None:
            loss = tag_loss
        else:
            raise ValueError('Both lm_loss and tag_loss are None.')
        
        return loss, lm_loss, tag_loss, entropy_loss, qt_ratio

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        lm_loss_sum = utils.item(sum(log.get('lm_loss', 0) for log in logging_outputs))
        tag_loss_sum = utils.item(sum(log.get('tag_loss', 0) for log in logging_outputs))
        entropy_loss_sum = utils.item(sum(log.get('entropy_loss', 0) for log in logging_outputs))

        lm_nll_loss_sum = utils.item(sum(log.get('lm_nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        n_z_tokens = utils.item(sum(log.get('n_z_tokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        qt_ratio = utils.item(sum(log.get('qt_ratio', 0) for log in logging_outputs))
        metrics.log_scalar('qt_ratio', qt_ratio / sample_size, sample_size, round=3)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        norm_loss_in_sample = logging_outputs[0]['norm_loss_in_sample']
        if norm_loss_in_sample:
            # loss has been normalized by number of tokens in the sample
            # only need to be normalized by number of samples in the batch
            metrics.log_scalar('lm_loss', lm_loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('tag_loss', tag_loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('entropy_loss', entropy_loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('lm_nll_loss', lm_nll_loss_sum / sample_size / math.log(2), ntokens, round=3)
            metrics.log_derived('lm_ppl', lambda meters: utils.get_perplexity(meters['lm_nll_loss'].avg))
        else:
            metrics.log_scalar('lm_loss', lm_loss_sum / ntokens / math.log(2), sample_size, round=3)
            metrics.log_scalar('tag_loss', tag_loss_sum / n_z_tokens / math.log(2), sample_size, round=3)
            metrics.log_scalar('entropy_loss', entropy_loss_sum / n_z_tokens / math.log(2), sample_size, round=3)
            metrics.log_scalar('lm_nll_loss', lm_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('lm_ppl', lambda meters: utils.get_perplexity(meters['lm_nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('bce_tagging_generation_joint_criterion')
class BCETaggingGenerationJointCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.lm_criterion = GuidedLabelSmoothedCrossEntropyCriterion(args, task)
        self.tag_criterion = GuidedLabelSmoothedBCEforTagging(args, task)
        self.tag_coef = args.tag_coef
        self.tag_only = args.tag_only
        self.norm_loss_in_sample = args.norm_loss_in_sample

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--tag-coef', default=1.0, type=float,
                            help='coefficient to combine tagging loss, \
                                0 means no tagging loss, and 1 means no generation loss.')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing in generation, 0 means no label smoothing')
        parser.add_argument('--tag-label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing in tagging, 0 means no label smoothing')
        parser.add_argument('--norm-loss-in-sample', action='store_true', default=False,
                            help='get token-level loss and set sample_size to 1.')
        parser.add_argument('--tag-only', action='store_true', default=False,
                            help='train only a tagger using only tagging loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], tag_only=self.tag_only)
        loss, lm_loss, tag_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        assert not self.args.sentence_avg
        
        if self.norm_loss_in_sample:
            sample_size = 1
        else:
            sample_size = sample['ntokens'] + sample['n_z_tokens']
        
        logging_output = {
            'loss': loss.data,
            'lm_loss': lm_loss.data if lm_loss is not None else 0.0,
            'tag_loss': tag_loss.data if tag_loss is not None else 0.0,
            # 'nll_loss': lm_nll_loss.data,
            'ntokens': sample['ntokens'],
            'n_z_tokens': sample['n_z_tokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'norm_loss_in_sample': self.norm_loss_in_sample,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # assert len(net_output)==3, f'net_output has {len(net_output)} elements'
        if self.tag_only:
            tag_scores = net_output
        else:
            x, extra, tag_scores = net_output
        
        lm_loss, tag_loss = None, None
        if self.tag_coef > 0.0:
            tag_loss, _ = self.tag_criterion.compute_loss(model, tag_output=[tag_scores,], sample=sample, reduce=reduce)
            if self.norm_loss_in_sample:
                tag_loss = tag_loss / sample['n_z_tokens']
        
        if not self.tag_only:
            lm_loss, _ = self.lm_criterion.compute_loss(model, net_output=[x, extra], sample=sample, reduce=reduce)
            if self.norm_loss_in_sample:
                lm_loss = lm_loss / sample['ntokens']
        
        if lm_loss is not None and tag_loss is not None:
            loss = lm_loss + self.tag_coef * tag_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif tag_loss is not None:
            loss = tag_loss
        else:
            raise ValueError('Both lm_loss and tag_loss are None.')
        
        return loss, lm_loss, tag_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        lm_loss_sum = utils.item(sum(log.get('lm_loss', 0) for log in logging_outputs))
        tag_loss_sum = utils.item(sum(log.get('tag_loss', 0) for log in logging_outputs))

        lm_nll_loss_sum = utils.item(sum(log.get('lm_nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        n_z_tokens = utils.item(sum(log.get('n_z_tokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        norm_loss_in_sample = logging_outputs[0]['norm_loss_in_sample']
        if norm_loss_in_sample:
            # loss has been normalized by number of tokens in the sample
            # only need to be normalized by number of samples in the batch
            metrics.log_scalar('lm_loss', lm_loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('tag_loss', tag_loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('lm_nll_loss', lm_nll_loss_sum / sample_size / math.log(2), ntokens, round=3)
            metrics.log_derived('lm_ppl', lambda meters: utils.get_perplexity(meters['lm_nll_loss'].avg))
        else:
            metrics.log_scalar('lm_loss', lm_loss_sum / ntokens / math.log(2), sample_size, round=3)
            metrics.log_scalar('tag_loss', tag_loss_sum / n_z_tokens / math.log(2), sample_size, round=3)
            metrics.log_scalar('lm_nll_loss', lm_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('lm_ppl', lambda meters: utils.get_perplexity(meters['lm_nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
