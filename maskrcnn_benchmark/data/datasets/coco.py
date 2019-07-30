# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import skimage.measure
import os
import sys
import math
import pycocotools.mask as mask_utils
import numpy as np
import skimage.transform
import scipy.ndimage
import cv2
import random
from PIL import Image
import warnings

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.utils import visualize_datasets


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, data_aug=False, is_train=False
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        self.use_mask = True
        self.data_aug = data_aug
        self.is_train = is_train

        print('min_area', np.array([ann['area'] for k, ann in self.coco.anns.items() if ann['iscrowd'] == 0]).min())

        if not self.use_mask:
            # convert rle to polygons
            for k, ann in self.coco.anns.items():
                if isinstance(ann['segmentation'], dict):
                    ann['segmentation'] = self.mask_to_polygon(ann['segmentation'])
            for k, v in self.coco.imgToAnns.items():
                for ann in v:
                    if isinstance(ann['segmentation'], dict):
                        ann['segmentation'] = self.mask_to_polygon(ann['segmentation'])

    def __getitem__(self, idx):
        # idx %= 1

        if self.use_mask:
            coco = self.coco
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anno = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']

            # filter crowd annotations
            # TODO might be better to add an extra field
            # anno = [obj for obj in anno if obj["iscrowd"] == 0]

            masks = [obj["segmentation"] for obj in anno]
            # RLE interpretation
            rle_sizes = [tuple(inst["size"]) for inst in masks]
            assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), (
                    "All the sizes must be the same size: %s" % rle_sizes
            )
            # in RLE, height come first in "size"
            rle_height, rle_width = rle_sizes[0]

            masks = mask_utils.decode(masks)  # [h, w, n]
            image = cv2.cvtColor(cv2.imread(os.path.join(self.root, path)), cv2.COLOR_BGR2RGB)

            if self.data_aug:
                image, window, scale, padding, crop = self.resize_image(
                    image,
                    min_dim=512,
                    max_dim=512,
                    min_scale=False,
                    mode='crop',
                    aspect_ratio=1.3,  # 1.5
                    zoom=1.5,  # 1.7
                    min_enlarge=1.2,  # 1.5
                )
                masks = self.resize_mask(masks, scale, padding, crop)

                if random.randint(0, 1):
                    image = np.ascontiguousarray(np.fliplr(image))
                    masks = np.ascontiguousarray(np.fliplr(masks))

                if random.randint(0, 1):
                    image = np.ascontiguousarray(np.flipud(image))
                    masks = np.ascontiguousarray(np.flipud(masks))

                ## Random rotation
                coin = np.random.random()
                if coin < 0.25:
                    k = 1
                elif (coin >= 0.25 and coin < 0.5):
                    k = 2
                elif (coin >= 0.5 and coin < 0.75):
                    k = 3
                else:
                    k = 0
                image = np.rot90(image, k=k, axes=(0, 1))
                masks = np.rot90(masks, k=k, axes=(0, 1))

                rot_range = 10.  # 22.5
                channel_shift_range = 15  # 20

                if np.random.uniform(0, 1) > 0.5:
                    image, masks = self.img_rot(image, masks, angle=np.random.uniform(-rot_range, rot_range))

                image = self.random_channel_shift(image, channel_shift_range, 2)

                # Note that some boxes might be all zeros if the corresponding mask got cropped out.
                # and here is to filter them out
                _idx = np.sum(masks, axis=(0, 1)) > 0
                masks = masks[:, :, _idx]
                # Bounding boxes. Note that some boxes might be all zeros
                # if the corresponding mask got cropped out.
                # bbox: [num_instances, (y1, x1, y2, x2)]
                boxes = self.extract_bboxes(masks)

                # visualize_datasets.vis_one_training_image(image, str(img_id),
                #                                           '/media/fs3017/eeum/nuclei/test',
                #                                           boxes, masks, is_box_xyxy=True)

                img = Image.fromarray(image)
                target = BoxList(torch.as_tensor(boxes), img.size, mode="xyxy")

                classes = [obj["category_id"] for obj in anno]
                classes = np.array([self.json_category_id_to_contiguous_id[c] for c in classes])[_idx]
                classes = torch.as_tensor(classes)
                target.add_field("labels", classes)

                is_crowd = np.array([obj["iscrowd"] for obj in anno])[_idx]
                is_crowd = torch.as_tensor(is_crowd)
                target.add_field("is_crowd", is_crowd)

                non_crowd_masks = masks[:, :, np.array([obj["iscrowd"] for obj in anno])[_idx]]
                centerness = scipy.ndimage.zoom(non_crowd_masks.max(axis=2), zoom=[0.25, 0.25], order=0)
                centerness = (centerness > 0).astype(np.float32)
                centerness[centerness == 0] = -1.
                centerness[centerness > 0] = 0.
                center_scale = 0.3
                gt_bbox = np.zeros(shape=(centerness.shape[0], centerness.shape[1], 4))
                anchor_bbox = np.zeros(shape=gt_bbox.shape)
                for xx in range(centerness.shape[1]):
                    for yy in range(centerness.shape[0]):
                        anchor_bbox[yy, xx, :] = [max(0.0, xx * 4 - 16), max(0.0, yy * 4 - 16),
                                                  min(xx * 4 + 16, masks.shape[1]), min(yy * 4 + 16, masks.shape[0])]
                for bi, box in enumerate(boxes):
                    if is_crowd[bi]:
                        continue
                    x, y, xe, ye = box
                    w = xe - x
                    h = ye - y
                    ctr_x = x * 0.25 + w * 0.25 * 0.5
                    ctr_y = y * 0.25 + h * 0.25 * 0.5
                    hw = w * 0.25 * 0.5 * center_scale
                    hh = h * 0.25 * 0.5 * center_scale
                    sx = math.floor(ctr_x - hw)
                    sy = math.floor(ctr_y - hh)
                    ex = max(sx + 1, math.ceil(ctr_x + hw))
                    ey = max(sy + 1, math.ceil(ctr_y + hh))
                    centerness[sy:ey, sx:ex] = 1.
                    gt_bbox[sy:ey, sx:ex, :] = [x, y, xe, ye]

                masks = torch.tensor(masks).permute(2, 0, 1)  # [n, h, w]
                assert masks.shape[1] == img.size[1]
                assert masks.shape[2] == img.size[0]
                masks = SegmentationMask(masks, img.size, mode='mask')
                target.add_field("masks", masks)

                if self._transforms is not None:
                    img, target = self._transforms(img, target)
            else:
                if self.is_train:
                    if random.randint(0, 1):
                        image = np.ascontiguousarray(np.fliplr(image))
                        masks = np.ascontiguousarray(np.fliplr(masks))

                    if random.randint(0, 1):
                        image = np.ascontiguousarray(np.flipud(image))
                        masks = np.ascontiguousarray(np.flipud(masks))

                # boxes = [obj["bbox"] for obj in anno]
                boxes = self.extract_bboxes(masks)

                # visualize_datasets.vis_one_training_image(image, str(img_id),
                #                                           '/media/fs3017/eeum/nuclei/test',
                #                                           boxes, masks, is_box_xyxy=False)

                img = Image.fromarray(image)
                target = BoxList(torch.as_tensor(boxes), img.size, mode="xyxy")

                classes = [obj["category_id"] for obj in anno]
                classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
                classes = torch.tensor(classes)
                target.add_field("labels", classes)

                is_crowd = [obj["iscrowd"] for obj in anno]
                is_crowd = torch.as_tensor(is_crowd)
                target.add_field("is_crowd", is_crowd)

                non_crowd_masks = masks[:, :, np.array([obj["iscrowd"] == 0 for obj in anno])]
                centerness = scipy.ndimage.zoom(non_crowd_masks.max(axis=2), zoom=[0.25, 0.25], order=0)
                centerness = (centerness > 0).astype(np.float32)
                centerness[centerness == 0] = -1.
                centerness[centerness > 0] = 0.
                center_scale = 0.3
                gt_bbox = np.zeros(shape=(centerness.shape[0], centerness.shape[1], 4))
                anchor_bbox = np.zeros(shape=gt_bbox.shape)
                for xx in range(centerness.shape[1]):
                    for yy in range(centerness.shape[0]):
                        anchor_bbox[yy, xx, :] = [max(0.0, xx * 4 - 16), max(0.0, yy * 4 - 16),
                                                  min(xx * 4 + 16, masks.shape[1]), min(yy * 4 + 16, masks.shape[0])]
                for bi, box in enumerate(boxes):
                    if is_crowd[bi]:
                        continue
                    x, y, xe, ye = box
                    w = xe - x
                    h = ye - y
                    ctr_x = x * 0.25 + w * 0.25 * 0.5
                    ctr_y = y * 0.25 + h * 0.25 * 0.5
                    hw = w * 0.25 * 0.5 * center_scale
                    hh = h * 0.25 * 0.5 * center_scale
                    sx = math.floor(ctr_x - hw)
                    sy = math.floor(ctr_y - hh)
                    ex = max(sx + 1, math.ceil(ctr_x + hw))
                    ey = max(sy + 1, math.ceil(ctr_y + hh))
                    centerness[sy:ey, sx:ex] = 1.
                    gt_bbox[sy:ey, sx:ex, :] = [x, y, xe, ye]
                    # print(gt_bbox[sy, sx, :], anchor_bbox[sy, sx, :])

                masks = torch.tensor(masks).permute(2, 0, 1)  # [n, h, w]
                assert masks.shape[1] == rle_height == img.size[1]
                assert masks.shape[2] == rle_width == img.size[0]
                masks = SegmentationMask(masks, img.size, mode='mask')
                target.add_field("masks", masks)

                target = target.clip_to_image(remove_empty=True)

                if self._transforms is not None:
                    img, target = self._transforms(img, target)

            # print(anchor_bbox, gt_bbox)
            return img, target, idx, \
                   (torch.as_tensor(centerness), torch.as_tensor(gt_bbox), torch.as_tensor(anchor_bbox))


        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def mask_to_polygon(self, mask, use_cv2 = True):
        reshaped_contour = []
        if use_cv2:
            contours, hierarchy = cv2_util.findContours(np.ascontiguousarray(mask_utils.decode(mask)),
                                                        cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                assert contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, contour
                contour = np.squeeze(contour, axis=1)
                # make it close
                if not np.all(contour[0, :] == contour[-1, :]):
                    contour = np.vstack([contour, contour[0, :]])
                reshaped_contour.append(contour.reshape(-1).tolist())
        else:
            contours = skimage.measure.find_contours(mask_utils.decode(mask), level=0.5)
            for contour in contours:
                assert contour.ndim == 2 and contour.shape[0] >= 3 and contour.shape[1] == 2, contour
                contour = np.flip(contour, axis=1)
                # make it close
                if not np.all(contour[0, :] == contour[-1, :]):
                    contour = np.vstack([contour, contour[0, :]])
                reshaped_contour.append(contour.astype(np.float32).reshape(-1).tolist())
        return reshaped_contour

    def resize_image(self, image, min_dim=512, max_dim=512, min_scale=False, mode="crop", aspect_ratio=1.3, zoom=1.5,
                     min_enlarge=1.2):
        """Resizes an image keeping the aspect ratio unchanged.

        aspect_ratio: 1.3, 1.5
        zoom: 1.5, 1.7
        min_enlarge: 1.2, 1.5
        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image
                of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None

        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        if zoom:
            ## Zooming and aspect ratio changes
            min_scale = min_dim / min(h, w)
            max_scale = max(min_enlarge,
                            min_scale * zoom)  ## We enlarge images at least by min_scale*zomm or min_enlarge
            scale = np.random.uniform(min_scale, max_scale)
            # scale = (scale,  scale*np.random.uniform(1,aspect_ratio)) ## change aspect ratio

            if np.random.uniform(0, 1) > 0.5:
                scale = (scale, scale * np.random.uniform(1, aspect_ratio))  ## change aspect ratio
            else:
                scale = (scale * np.random.uniform(1, aspect_ratio), scale)  ## change aspect ratio

        else:
            scale = (scale, scale)

        # Resize image using bilinear interpolation
        if scale != (1, 1):
            ## Note that my original submission used sdimage for resizing, I have changed it to skimage to be consistent with the latest Mask_RCNN
            image = skimage.transform.resize(
                image, (round(h * scale[0]), round(w * scale[1])),
                order=1, mode="constant", preserve_range=True)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            ## My bug in the final implementation resulted in y and x being always = 0 ...
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:
            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop

    def resize_mask(self, mask, scale, padding, crop=None):
        """Resizes a mask using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.

        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
        # Suppress warning from scipy 0.13.0, the output shape of zoom() is
        # calculated with round() instead of int()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = scipy.ndimage.zoom(mask, zoom=[scale[0], scale[1], 1], order=0)
        if crop is not None:
            y, x, h, w = crop
            mask = mask[y:y + h, x:x + w]
        else:
            mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    def crop_center(self, img_pre, xcrop, ycrop):
        ysize, xsize, chan = img_pre.shape
        xoff = (xsize - xcrop) // 2
        yoff = (ysize - ycrop) // 2
        # img= img_pre[yoff:-yoff,xoff:-xoff]
        img = img_pre[yoff:(yoff + ycrop), xoff:(xoff + xcrop)]
        return img

    #########################################################################
    ## Rotate image around a centerpoint and return transformation matrix
    #########################################################################
    def img_rot(self, img, msk, angle):
        ## center: (cX,xY) tuple with rotation center
        ## angle: Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).)
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH), None, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
        msk = cv2.warpAffine(msk, M, (nW, nH), None, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
        if len(msk.shape) == 2:
            msk = np.expand_dims(msk, 2)

        img = self.crop_center(img, h, w)
        msk = self.crop_center(msk, h, w)
        msk = np.clip(msk, 0, 1)

        # if len(img.shape)==2:
        #     img=np.expand_dims(img,2)
        return img, msk

    #########################################################################

    def random_channel_shift(self, x, intensity, channel_axis=0):
        image_dtype = x.dtype
        x = np.rollaxis(x, channel_axis, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                          for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1).astype(image_dtype)
        return x

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                assert False, (horizontal_indicies, vertical_indicies)
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)
