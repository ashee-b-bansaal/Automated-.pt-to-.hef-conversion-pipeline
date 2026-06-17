# Copy-paste examples

Assumptions:

```bash
export TOOLKIT=$HOME/hailo-edge-model-toolkit
export MODEL_ZOO_ROOT=$HOME/Video_streaming/hailo_model_zoo
export CALIB_DIR=$HOME/calibration_images
export HAILO_OUT=$HOME/hailo_outputs
```

## Existing pretrained ResNet-50 from Model Zoo

```bash
source ~/hailo_env/bin/activate
$TOOLKIT/scripts/compile_zoo_model.sh   --model resnet_v1_50   --model-zoo-root "$MODEL_ZOO_ROOT"   --calib-dir "$CALIB_DIR"   --hw-arch hailo8   --output-dir "$HAILO_OUT/resnet50_pretrained"
```

## Fine-tuned ResNet-50

Export in a separate PyTorch environment:

```bash
source ~/model_export_env/bin/activate
$TOOLKIT/hmct export-classifier   --arch resnet50   --checkpoint best.pt   --num-classes 2   --image-size 224   --output "$HOME/models/resnet50_finetuned.onnx"
$TOOLKIT/scripts/check_onnx.py "$HOME/models/resnet50_finetuned.onnx"
```

Compile in the Hailo environment:

```bash
source ~/hailo_env/bin/activate
$TOOLKIT/scripts/compile_zoo_model.sh   --model resnet_v1_50   --onnx "$HOME/models/resnet50_finetuned.onnx"   --classes 2   --model-zoo-root "$MODEL_ZOO_ROOT"   --calib-dir "$CALIB_DIR"   --hw-arch hailo8   --output-dir "$HAILO_OUT/resnet50_finetuned"
```

## Existing ViT from Model Zoo

```bash
source ~/hailo_env/bin/activate
$TOOLKIT/scripts/compile_zoo_model.sh   --model vit_small   --model-zoo-root "$MODEL_ZOO_ROOT"   --calib-dir "$CALIB_DIR"   --hw-arch hailo8   --output-dir "$HAILO_OUT/vit_small"
```

## Custom YOLOv8/YOLO11

```bash
source ~/model_export_env/bin/activate
$TOOLKIT/scripts/export_yolo_onnx.sh --weights best.pt --imgsz 640

source ~/hailo_env/bin/activate
$TOOLKIT/scripts/compile_zoo_model.sh   --model yolov8s   --onnx best.onnx   --classes 3   --model-zoo-root "$MODEL_ZOO_ROOT"   --calib-dir "$CALIB_DIR"   --output-dir "$HAILO_OUT/yolov8s_custom"
```

## YOLOv5 with dynamic NMS repair

The legacy YOLOv5 graph can require model-specific NMS layer-name repair:

```bash
source ~/hailo_env/bin/activate
MODEL_ZOO_ROOT="$MODEL_ZOO_ROOT" YOLOV5_DIR="$HOME/hailo_clean/yolov5" CALIB="$CALIB_DIR" OUT_DIR="$HAILO_OUT/yolov5" MODELS="yolov5s yolov5m" $TOOLKIT/scripts/convert_yolov5_dynamic.sh
```
