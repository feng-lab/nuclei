# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import random
import string
import os

from torchvision.utils import save_image

import numpy as np

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from maskrcnn_benchmark.layers.misc import interpolate
from maskrcnn_benchmark.utils.visualize_datasets import vis_one_training_image_with_pred


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        if cfg.MODEL.RPN.PRED_TARGETS:
            self.centerness_logits = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
            self.center_bbox_pred = nn.Conv2d(
                in_channels, 1 * 4, kernel_size=1, stride=1
            )

            for l in [self.centerness_logits, self.center_bbox_pred]:
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)
        else:
            self.centerness_logits = None

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = None
        center_bbox_reg = None
        for idx, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
            if idx == 0 and self.centerness_logits:
                centerness = self.centerness_logits(t)
                center_bbox_reg = self.center_bbox_pred(t)
        return logits, bbox_reg, centerness, center_bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        # rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        rpn_box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, attention_map=None, centerness_pack=None, iter=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression, centerness, rpn_center_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            boxes, losses, pred_targets = self._forward_train(anchors, objectness, rpn_box_regression, targets,
                                                              centerness, rpn_center_box_regression, centerness_pack)
        else:
            boxes, losses, pred_targets = self._forward_test(anchors, objectness, rpn_box_regression,
                                                             centerness, rpn_center_box_regression)

        if False and self.training and iter % 50 == 0 and iter <= 1000:
            self.visualize_attentions(images, targets, attention_map, iter, centerness_pack, centerness,
                                      pred_targets)

        return boxes, losses, pred_targets

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets,
                       centerness, rpn_center_box_regression, centerness_pack):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
            pred_targets = None
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes, pred_targets = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets,
                    centerness, rpn_center_box_regression, centerness_pack
                )
        loss_objectness, loss_rpn_box_reg, loss_centerness, loss_rpn_center_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets,
            pred_targets, centerness, rpn_center_box_regression, centerness_pack
        )
        if self.cfg.MODEL.RPN.PRED_TARGETS:
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_centerness": loss_centerness,
                "loss_rpn_center_box_reg": loss_rpn_center_box_reg,
            }
        else:
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses, pred_targets

    def _forward_test(self, anchors, objectness, rpn_box_regression,
                      centerness, rpn_center_box_regression):
        boxes, pred_targets = self.box_selector_test(anchors, objectness, rpn_box_regression, None,
                                                     centerness, rpn_center_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}, pred_targets

    def visualize_attentions(self, images, targets, attention_map, iter, centerness_pack, centerness, pred_targets):
        for image, target, coeff, attention, cp, center, pt in zip(images.tensors, targets, attention_map[0],
                                                                   attention_map[1],
                                                                   centerness_pack, centerness, pred_targets):
            gt_centerness, gt_bbox, anchor_bbox = cp
            masks = target.get_field("masks").instances.masks
            masks = masks[:masks.shape[0],:,:]
            folder = '/media/fs3017/eeum/nuclei/test'
            folder = '/home/feng/data/test'
            name = str(iter) + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            image_name = os.path.join(folder, name + '_image.tif')
            mask_name = os.path.join(folder, name + '_masks.tif')
            attention_name = os.path.join(folder,  name + '_attention.tif')
            print(image.shape, masks.shape, coeff.shape, attention.shape)
            # print(image.device, masks.device, attention[0].device)
            mask, _ = masks.max(dim=0)
            mask = mask.to(attention.device)
            _, ih, iw = image.shape
            pad_mask = mask.new(ih, iw).zero_()
            h, w = mask.shape
            pad_mask[:h, :w].copy_(mask)
            resized_masks = interpolate(
                input=pad_mask[None, None].float(),
                size=(ih//4, iw//4),
                mode="bilinear",
                align_corners=False,
            )[0]
            # print(resized_masks.shape)
            # att = (resized_masks.reshape(1, ih//4*iw//4) @ attention).reshape(1, 1, ih//4, iw//4)
            att = (resized_masks.reshape(1, (ih//4)*(iw//4)) @ coeff.transpose(0, 1) @ attention).reshape(1, 1, ih//4, iw//4)
            att = (att - att.min()) / (att.max() - att.min())
            # att[att < 0.5] = 0.0
            save_image(att, attention_name, normalize=True)
            if pt:
                im = image.detach().cpu().numpy()
                im = (im - im.min()) / (im.max() - im.min())
                im *= 255.
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                # print(pt.bbox)
                vis_one_training_image_with_pred(im, image_name, folder, pt.bbox)
            else:
                save_image(image[None], image_name, normalize=True)
            save_image(resized_masks[None], mask_name, normalize=True)
            save_image(center[None], os.path.join(folder, name + '_center.tif'), normalize=True)
            save_image(gt_centerness[None, None], os.path.join(folder, name + '_center_gt.tif'), normalize=True)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
