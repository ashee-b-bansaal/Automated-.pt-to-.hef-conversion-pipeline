# Hailo-8 Model Zoo inventory

This checked-in list is a convenience snapshot of the public Hailo-8
classification and object-detection tables consulted on **2026-06-17**. The
installed Model Zoo checkout is the source of truth because recipes differ by
release.

Generate the exact local inventory:

```bash
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo --task classification
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo --pattern vit
```

A listed model means a recipe/pretrained artifact exists for that release. It
does not guarantee that an arbitrarily modified ONNX graph will work with the
same recipe.

## Classification

- `cas_vit_m`
- `cas_vit_s`
- `cas_vit_t`
- `davit_tiny`
- `deit_base`
- `deit_small`
- `deit_tiny`
- `efficientformer_l1`
- `efficientnet_l`
- `efficientnet_lite0`
- `efficientnet_lite1`
- `efficientnet_lite2`
- `efficientnet_lite3`
- `efficientnet_lite4`
- `efficientnet_m`
- `efficientnet_s`
- `fastvit_sa12`
- `hardnet39ds`
- `hardnet68`
- `inception_v1`
- `levit128`
- `levit192`
- `levit256`
- `levit384`
- `mobilenet_v1`
- `mobilenet_v2_1.0`
- `mobilenet_v2_1.4`
- `mobilenet_v3`
- `regnetx_1.6gf`
- `regnetx_800mf`
- `repghost_1_0x`
- `repghost_2_0x`
- `repvgg_a1`
- `repvgg_a2`
- `resmlp12_relu`
- `resnet_v1_18`
- `resnet_v1_34`
- `resnet_v1_50`
- `resnext26_32x4d`
- `resnext50_32x4d`
- `squeezenet_v1.1`
- `swin_small`
- `swin_tiny`
- `vit_base`
- `vit_base_bn`
- `vit_small`
- `vit_small_bn`
- `vit_tiny`
- `vit_tiny_bn`

### Relevant families

- **ResNet:** `resnet_v1_18`, `resnet_v1_34`, `resnet_v1_50`
- **Vision Transformers:** `vit_tiny`, `vit_small`, `vit_base` and BN variants;
  DeiT, Swin, LeViT, DaViT, CAS-ViT, FastViT are also represented.
- **Mobile classifiers:** MobileNet, EfficientNet Lite, RepGhost, SqueezeNet.
- **ViT-L/14 is not in this Hailo-8 public classification snapshot.** Exporting
  it is possible to attempt, but compilation is a BYOM experiment rather than
  a known-good Zoo path.

## Object detection

- `centernet_resnet_v1_18_postprocess`
- `centernet_resnet_v1_50_postprocess`
- `damoyolo_tinynasL20_T`
- `damoyolo_tinynasL25_S`
- `damoyolo_tinynasL35_M`
- `detr_resnet_v1_18_bn`
- `efficientdet_lite0`
- `efficientdet_lite1`
- `efficientdet_lite2`
- `nanodet_repvgg`
- `nanodet_repvgg_a12`
- `nanodet_repvgg_a1_640`
- `ssd_mobilenet_v1`
- `ssd_mobilenet_v2`
- `tiny_yolov3`
- `tiny_yolov4`
- `yolo26m`
- `yolo26n`
- `yolo26s`
- `yolov10b`
- `yolov10n`
- `yolov10s`
- `yolov10x`
- `yolov11l`
- `yolov11m`
- `yolov11n`
- `yolov11s`
- `yolov11x`
- `yolov3`
- `yolov3_416`
- `yolov3_gluon`
- `yolov3_gluon_416`
- `yolov4_leaky`
- `yolov5m`
- `yolov5m6_6.1`
- `yolov5m_6.1`
- `yolov5m_wo_spp`
- `yolov5s`
- `yolov5s_bbox_decoding_only`
- `yolov5s_c3tr`
- `yolov5s_wo_spp`
- `yolov5xs_wo_spp`
- `yolov5xs_wo_spp_nms_core`
- `yolov6n`
- `yolov6n_0.2.1`
- `yolov6n_0.2.1_nms_core`
- `yolov7`
- `yolov7_tiny`
- `yolov7e6`
- `yolov8l`
- `yolov8m`
- `yolov8n`
- `yolov8s`
- `yolov8s_bbox_decoding_only`
- `yolov8x`
- `yolov9c`
- `yolox_l_leaky`
- `yolox_s_leaky`
- `yolox_s_wide_leaky`
- `yolox_tiny`

## Other tasks

The Model Zoo also contains release-dependent recipes for segmentation, pose,
depth, OCR, face detection/landmarks, re-identification, video classification,
and zero-shot classification. Rather than keeping a stale hardcoded list, use
`hmct zoo-list` against the installed v2.x checkout.
