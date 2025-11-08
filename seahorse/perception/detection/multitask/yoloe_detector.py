#!/usr/bin/env python

# Copyright 2025 WheelOS. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2025-11-06
# Author: daohu527


import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .base_detector import ObjectDetector
from .contracts import BoundingBox, DetectionResult, DetectionResults


class YOLOESegDetector(ObjectDetector):
    def __init__(
        self,
        model_path: str,
        class_names: Dict[int, str],
        img_size: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        if not class_names or not isinstance(class_names, dict):
            raise ValueError("`class_names` It must be a non-empty dictionary.")

        self.model_path = model_path
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_coeffs = 32

        np.random.seed(42)
        colors_raw = np.random.randint(
            0, 255, size=(self.num_classes, 3), dtype=np.uint8
        )
        self.colors = {
            name: color.tolist()
            for name, color in zip(self.class_names.values(), colors_raw)
        }

        super().__init__(device)

    def _load_model(self) -> torch.nn.Module:
        print(f"INFO: Loading TorchScript model from '{self.model_path}'...")
        model = torch.jit.load(self.model_path, map_location=self.device)
        return model.eval()

    def _preprocess(self, image: np.ndarray) -> (torch.Tensor, tuple, tuple):
        image_rgb = image[..., ::-1]
        h0, w0 = image_rgb.shape[:2]
        original_shape = (h0, w0)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .to(self.device)
            .float()
            .permute(2, 0, 1)
        )
        tensor /= 255.0
        r = self.img_size / max(h0, w0)
        if r != 1:
            new_h, new_w = int(h0 * r), int(w0 * r)
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        h, w = tensor.shape[1:]
        dh, dw = self.img_size - h, self.img_size - w
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        padded_tensor = F.pad(
            tensor, (left, right, top, bottom), mode="constant", value=114 / 255.0
        )
        model_input_shape = (self.img_size, self.img_size)
        return padded_tensor.unsqueeze(0), original_shape, model_input_shape

    def _non_max_suppression(
        self, boxes: torch.Tensor, scores: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        max_coord = boxes.max()
        offsets = class_indices.to(boxes.dtype) * (max_coord + 1)
        boxes_for_nms = boxes + offsets[:, None]
        x1, y1, x2, y2 = (
            boxes_for_nms[:, 0],
            boxes_for_nms[:, 1],
            boxes_for_nms[:, 2],
            boxes_for_nms[:, 3],
        )
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            inter_w = torch.clamp(xx2 - xx1, min=0.0)
            inter_h = torch.clamp(yy2 - yy1, min=0.0)
            inter = inter_w * inter_h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = torch.where(ovr <= self.iou_thresh)[0]
            order = order[inds + 1]
        return (
            torch.tensor(keep, dtype=torch.long, device=boxes.device)
            if keep
            else torch.tensor([], dtype=torch.long)
        )

    def _xywh2xyxy(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def _scale_coords(
        self, model_input_shape: tuple, coords: torch.Tensor, original_shape: tuple
    ):
        h_model, w_model = model_input_shape
        h_orig, w_orig = original_shape
        gain = min(h_model / h_orig, w_model / w_orig)
        pad_w, pad_h = (w_model - w_orig * gain) / 2, (h_model - h_orig * gain) / 2
        coords[:, [0, 2]] -= pad_w
        coords[:, [1, 3]] -= pad_h
        coords[:, :4] /= gain
        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, w_orig)
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, h_orig)

    def _process_masks(
        self,
        low_res_masks: torch.Tensor,
        model_input_shape: tuple,
        original_shape: tuple,
    ) -> np.ndarray:
        h_model, w_model = model_input_shape
        h_orig, w_orig = original_shape
        gain = min(h_model / h_orig, w_model / w_orig)
        pad_w, pad_h = (w_model - w_orig * gain) / 2, (h_model - h_orig * gain) / 2
        top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
        bottom, right = int(round(h_model - pad_h + 0.1)), int(
            round(w_model - pad_w + 0.1)
        )
        masks_upsampled = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(h_model, w_model),
            mode="bilinear",
            align_corners=False,
        )
        masks_cropped = masks_upsampled[:, :, top:bottom, left:right]
        final_masks = F.interpolate(
            masks_cropped, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        ).squeeze(1)
        final_masks = torch.sigmoid(final_masks)
        return (final_masks > 0.5).cpu().numpy().astype(np.uint8)

    def _postprocess(
        self, outputs: tuple, original_shape: tuple, model_input_shape: tuple
    ) -> DetectionResults:

        det_seg_output, mask_prototypes = outputs

        actual_feature_dim = det_seg_output.shape[1]
        expected_feature_dim = 4 + self.num_classes + self.num_coeffs

        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f"Dimensionality mismatch! Model output {actual_feature_dim} However, the program is expected to {expected_feature_dim} Dimension (4 + {self.num_classes} + {self.num_coeffs})."
            )

        preds = det_seg_output.squeeze(0).T
        mask_prototypes = mask_prototypes.squeeze(0)

        boxes, class_scores, mask_coeffs = torch.split(
            preds, [4, self.num_classes, self.num_coeffs], dim=1
        )

        scores, class_indices = class_scores.max(dim=1)
        keep = scores > self.conf_thresh

        if not torch.any(keep):
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        class_indices = class_indices[keep]
        mask_coeffs = mask_coeffs[keep]

        boxes_xyxy = self._xywh2xyxy(boxes)
        keep_indices = self._non_max_suppression(boxes_xyxy, scores, class_indices)

        if not keep_indices.numel():
            return []

        final_boxes = boxes_xyxy[keep_indices]
        final_scores = scores[keep_indices]
        final_classes = class_indices[keep_indices]
        final_mask_coeffs = mask_coeffs[keep_indices]

        proto_h, proto_w = mask_prototypes.shape[1], mask_prototypes.shape[2]
        low_res_masks = torch.matmul(
            final_mask_coeffs, mask_prototypes.view(self.num_coeffs, -1)
        )
        low_res_masks = low_res_masks.view(-1, proto_h, proto_w)
        final_masks_np = self._process_masks(
            low_res_masks, model_input_shape, original_shape
        )
        self._scale_coords(model_input_shape, final_boxes, original_shape)

        results = []
        for i in range(final_boxes.shape[0]):
            box = final_boxes[i].cpu().numpy().astype(int)
            score = final_scores[i].item()
            class_id = final_classes[i].item()
            label = self.class_names[class_id]
            color = self.colors[label]
            mask = final_masks_np[i]

            results.append(
                DetectionResult(
                    bounding_box=BoundingBox(box[0], box[1], box[2], box[3]),
                    class_id=class_id,
                    label=label,
                    score=score,
                    color=color,
                    mask=mask,
                )
            )
        return results

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> DetectionResults:
        input_tensor, original_shape, model_input_shape = self._preprocess(image)
        model_outputs = self.model(input_tensor)
        outputs_on_cpu = [o.cpu() for o in model_outputs]
        results = self._postprocess(outputs_on_cpu, original_shape, model_input_shape)
        return results
