# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

import scipy.ndimage
import numpy as np


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func, pred_targets_as_true=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = ["is_crowd"]
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds', 'crowd']
        self.center_sampler = BalancedPositiveNegativeSampler(512, 0.5)

        self.pred_targets_as_true = pred_targets_as_true

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        # matched_vals, matches = match_quality_matrix.max(dim=1)
        # assert (matched_vals >= 0.7).all(), ((matched_vals >= 0.7).sum(), matched_vals.shape[0], matched_vals.mean(), matched_vals, anchor,
        #                                      anchor.bbox,
        #                                      target, target.bbox)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets, pred_targets=None):
        labels = []
        regression_targets = []
        labels_for_objectness = []
        for imgidx, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # discard indices that are between thresholds
            if "crowd" in self.discard_cases:
                inds_to_discard = (matched_targets.get_field("is_crowd") > 0) & (matched_idxs >= 0)
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            # ignore anchors that match pred_targets
            if pred_targets and pred_targets[imgidx]:
                matched_pred_targets = self.match_targets_to_anchors(
                    anchors_per_image, pred_targets[imgidx]
                )
                matched_pred_idxs = matched_pred_targets.get_field("matched_idxs")
                if self.pred_targets_as_true:
                    labels_per_image_obj = labels_per_image.clone()
                    labels_per_image_obj[matched_pred_idxs >= 0] = 1
                    inds_to_discard = bg_indices & \
                                      (matched_pred_idxs == Matcher.BETWEEN_THRESHOLDS)
                    labels_per_image_obj[inds_to_discard] = -1
                    labels_for_objectness.append(labels_per_image_obj)
                else:
                    inds_to_discard = bg_indices & \
                                      ((matched_pred_idxs == Matcher.BETWEEN_THRESHOLDS) | (matched_pred_idxs >= 0))
                labels_per_image[inds_to_discard] = -1
            else:
                if self.pred_targets_as_true:
                    labels_for_objectness.append(labels_per_image.clone())

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets, labels_for_objectness


    def __call__(self, anchors, objectness, box_regression, targets,
                 pred_targets, centerness, rpn_center_box_regression, centerness_pack):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        centerness_loss = None
        rpn_center_box_loss = None
        if pred_targets:
            labels = []
            regression_targets = []
            centers = []
            box_regressions = []
            for center_target, img_centerness, center_box_reg in \
                    zip(centerness_pack, centerness, rpn_center_box_regression):
                gt_centerness, gt_bbox, anchor_bbox = center_target
                labels.append(gt_centerness.reshape(-1))
                img_centerness = img_centerness[0, :gt_centerness.shape[0], :gt_centerness.shape[1]]
                centers.append(img_centerness.reshape(-1))
                center_box_reg = center_box_reg[:, :gt_centerness.shape[0], :gt_centerness.shape[1]].permute(1, 2, 0).reshape(-1, 4)
                box_regressions.append(center_box_reg)
                regression_targets.append(self.box_coder.encode(gt_bbox.view(-1, 4), anchor_bbox.view(-1, 4)))
                # print(gt_centerness.shape)
                # print(center.shape)
                # print(center_box_reg.shape)
                # print(gt_bbox.shape)
                # print(anchor_bbox.shape)
            sampled_pos_inds, sampled_neg_inds = self.center_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)
            box_regressions = torch.cat(box_regressions, dim=0)
            centers = torch.cat(centers, dim=0)
            rpn_center_box_loss = smooth_l1_loss(
                box_regressions[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())
            centerness_loss = F.binary_cross_entropy_with_logits(
                centers[sampled_inds], labels[sampled_inds]
            )

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, labels_for_objectness = self.prepare_targets(anchors, targets, pred_targets)
        if self.pred_targets_as_true:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels_for_objectness)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

            objectness = objectness.squeeze()

            labels_for_objectness = torch.cat(labels_for_objectness, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels_for_objectness[sampled_inds]
            )

            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())
        else:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness, box_regression = \
                    concat_box_prediction_layers(objectness, box_regression)

            objectness = objectness.squeeze()

            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())

            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds]
            )

        return objectness_loss, box_loss, centerness_loss, rpn_center_box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels,
        pred_targets_as_true=cfg.MODEL.RPN.PRED_TARGETS_AS_TRUE,
    )
    return loss_evaluator
