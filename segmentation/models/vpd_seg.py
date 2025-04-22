import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from vpd import UNetWrapper, TextAdapter, EncoderControlNet 

import copy


@SEGMENTORS.register_module()
class VPDSeg(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 decode_head,
                 base_size = 512,
                 sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
                 sd_config = "/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/v1-inference.yaml",
                 unet_config=dict(),
                 class_embedding_path='/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 max_boxes = 6,
                 **args):
        super().__init__(init_cfg)
        config = OmegaConf.load(sd_config)
        config.model.params.ckpt_path = f'{sd_path}'
        config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'
        
        # prepare the unet        
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        unet_a = sd_model.model
        self.unet = UNetWrapper(unet_a, base_size=base_size, **unet_config)
        self.box_encoder = EncoderControlNet(in_channels = max_boxes )
        
        sd_model.model = None
        sd_model.first_stage_model = None
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        self.sd_model = sd_model

        # class embeddings & text adapter
        class_embeddings = torch.load(class_embedding_path)
        self.register_buffer('class_embeddings', class_embeddings)
        text_dim = class_embeddings.size(-1)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, boxes=None):
        """Extract features from images and apply control if boxes are provided."""
        with torch.no_grad():
            latents = self.encoder_vq.encode(img)
        latents = latents.mode().detach()
        # Get box-derived control features
        
        if boxes is not None:
            # box_map = self.make_box_map(boxes, img.shape[-2:], img.device)
            box_feats = self.box_encoder(boxes)
        else:
            box_feats = None

        # Cross-attention conditioning
        c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma)
        t = torch.ones((img.shape[0],), device=img.device).long()
        
        # Send latents, context, and control to UNet

        outs = self.unet(latents, t, context=c_crossattn, box_control=box_feats)

        return outs
    
    def make_box_map(self, boxes, img_size, device):
        """Convert bounding boxes to binary control maps (1 channel)."""
        B, H, W = len(boxes), *img_size
        maps = torch.zeros((B, 1, H, W), device=device)

        for i, bboxes in enumerate(boxes):  # boxes[i] = [N, 4] in xyxy format
            for box in bboxes:
                x1, y1, x2, y2 = box.int()
                maps[i, 0, y1:y2, x1:x2] = 1.0
        return maps
    
    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg, gt_bbox_masks=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img, boxes=gt_bbox_masks)

        if self.with_neck:
            x = self.neck(x)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def encode_decode(self, img, img_metas, gt_bbox_masks=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, boxes=gt_bbox_masks)
        if self.with_neck:
            x = list(self.neck(x))
        out = self._decode_head_forward_test(x, img_metas)  
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, gt_bbox_masks=None, rescale=True):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta,gt_bbox_masks=gt_bbox_masks)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, gt_bbox_masks= None,rescale = True):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        if isinstance(img_meta[0], list):
            img_meta = [m for meta in img_meta for m in meta]
        assert self.test_cfg.mode in ['slide', 'whole']

        if self.test_cfg.mode == 'slide':
            seg_logits = self.whole_inference(img, img_meta, gt_bbox_masks=gt_bbox_masks, rescale=rescale)
        else:
            seg_logits = self.whole_inference(img, img_meta, gt_bbox_masks=gt_bbox_masks, rescale=rescale)

        seg_logits = F.softmax(seg_logits, dim=1)
        batch_preds = []
        for i in range(seg_logits.shape[0]):
            pred = seg_logits[i:i+1]  # shape: (1, C, H, W)
            meta = img_meta[i]

            if meta.get('flip', False):
                flip_direction = meta.get('flip_direction', 'horizontal')
                if flip_direction == 'horizontal':
                    pred = pred.flip(dims=(3,))
                elif flip_direction == 'vertical':
                    pred = pred.flip(dims=(2,))
            batch_preds.append(pred)

        return torch.cat(batch_preds, dim=0)  # (B, C, H, W)

    def simple_test(self, img, img_meta, gt_bbox_masks= None,rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, gt_bbox_masks=gt_bbox_masks,rescale= rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred