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
from typing import List

from .base_detector import ObjectDetector
from .contracts import BoundingBox, DetectionResult, DetectionResults


class Detector(ObjectDetector):
    """
    A detector using native PyTorch for YOLO model inference, without relying on ultralytics or cv2.
    The model must be exported in TorchScript format.
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 640,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        super().__init__(device)
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        np.random.seed(42)
        # Generate a random BGR color for each class
        colors = np.random.randint(
            0, 255, size=(len(self.class_names), 3), dtype=np.uint8
        )
        self.colors = {
            name: color.tolist() for name, color in zip(self.class_names, colors)
        }

    def _load_model(self) -> torch.nn.Module:
        """Load the model from a TorchScript file."""
        print(f"INFO: Loading TorchScript model from '{self.model_path}'...")
        model = torch.jit.load(self.model_path, map_location=self.device)
        return model.eval()

    def _preprocess(self, image: np.ndarray) -> (torch.Tensor, tuple):
        """Preprocess the NumPy image into model input format, entirely using PyTorch operations."""
        h0, w0 = image.shape[:2]
        tensor = torch.from_numpy(image).to(self.device).float().permute(2, 0, 1)
        tensor = tensor.flip(0).unsqueeze(0)
        tensor /= 255.0

        r = self.img_size / max(h0, w0)
        if r != 1:
            new_h, new_w = int(h0 * r), int(w0 * r)
            mode = "area" if r < 1 else "bilinear"
            if mode == "area":
                tensor = F.interpolate(tensor, size=(new_h, new_w), mode=mode)
            else:
                tensor = F.interpolate(
                    tensor, size=(new_h, new_w), mode=mode, align_corners=False
                )

        h, w = tensor.shape[2:]
        dh, dw = self.img_size - h, self.img_size - w
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        tensor = F.pad(
            tensor, (left, right, top, bottom), mode="constant", value=114 / 255.0
        )

        return tensor, (h0, w0)

    def _non_max_suppression(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Pure PyTorch implementation of Non-Maximum Suppression (NMS)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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

            inter = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = torch.where(ovr <= self.iou_thresh)[0]
            order = order[inds + 1]

        return (
            torch.stack(keep)
            if keep
            else torch.tensor([], dtype=torch.long, device=boxes.device)
        )

    def _postprocess(
        self, preds: torch.Tensor, original_shape: tuple
    ) -> DetectionResults:
        """
        Post-process the model’s raw output.
        This implementation assumes the model outputs a tensor of shape (batch, 4 + num_classes, num_proposals),
        e.g., (1, 84, 8400).
        """
        # 1. Ensure the shape is correct and transpose: (1, 84, 8400) -> (8400, 84)
        preds = preds.squeeze(0).T

        # 2. Separate boxes and class scores from predictions
        boxes, class_scores = preds[:, :4], preds[:, 4:]

        # 3. Find best class and corresponding confidence for each predicted box
        scores, class_indices = torch.max(class_scores, dim=1)

        # 4. Filter by confidence threshold
        keep = scores > self.conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        class_indices = class_indices[keep]

        if not boxes.shape[0]:
            return []

        # 5. Convert box format (center_x, center_y, width, height) -> (x1, y1, x2, y2)
        boxes = self._xywh2xyxy(boxes)

        # 6. Execute category-wise NMS (Multi-class NMS)
        # By adding a large coordinate offset for different classes so NMS can treat each class independently
        max_coord = boxes.max()
        offsets = class_indices.to(boxes.dtype) * (max_coord + 1)
        boxes_for_nms = boxes + offsets[:, None]

        keep_indices = self._non_max_suppression(boxes_for_nms, scores)

        # 7. Final filtering according to NMS results
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_indices = class_indices[keep_indices]

        # 8. Scale coordinates from model input size back to original image size
        model_input_shape = (self.img_size, self.img_size)
        self._scale_coords(model_input_shape, boxes, original_shape)

        # 9. Build return results
        results = []
        for box, score, cls in zip(boxes, scores, class_indices):
            class_id = int(cls)
            label = self.class_names[class_id]
            color = self.colors[label]  # get color from mapping

            results.append(
                DetectionResult(
                    bounding_box=BoundingBox(
                        int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    ),
                    class_id=class_id,
                    label=label,
                    score=float(score),
                    color=color,  # add color to the result object
                )
            )
        return results

    def _xywh2xyxy(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (center_x, center_y, width, height) format to (x1, y1, x2, y2) format."""
        y = torch.empty_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def _scale_coords(
        self, model_input_shape: tuple, coords: torch.Tensor, original_shape: tuple
    ):
        """Scale coordinates from the model input size (letter-boxed) back to the original image size."""
        h_model, w_model = model_input_shape
        h_orig, w_orig = original_shape

        gain = min(h_model / h_orig, w_model / w_orig)
        pad_w, pad_h = (w_model - w_orig * gain) / 2, (h_model - h_orig * gain) / 2

        coords[:, [0, 2]] -= pad_w
        coords[:, [1, 3]] -= pad_h
        coords[:, :4] /= gain

        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, w_orig)
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, h_orig)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> DetectionResults:
        """Implement the standard ‘preprocess -> infer -> postprocess’ pipeline."""
        input_tensor, original_shape = self._preprocess(image)

        model_outputs = self.model(input_tensor)
        if isinstance(model_outputs, tuple):
            model_outputs = model_outputs[0]

        results = self._postprocess(model_outputs, original_shape)

        return results
