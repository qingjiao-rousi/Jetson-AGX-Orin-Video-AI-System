#!/usr/bin/env bash
# ─── 模型获取脚本 (在 Jetson AGX Orin 上执行) ───
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODEL_DIR"

echo "=== Step 1: 下载 YOLOv8n PyTorch 权重 ==="
cd "$MODEL_DIR"
if [ ! -f yolov8n.pt ]; then
    wget -O yolov8n.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    echo "  ✓ yolov8n.pt 下载完成"
else
    echo "  ✓ yolov8n.pt 已存在, 跳过"
fi

echo ""
echo "=== Step 2: 安装 ultralytics ==="
pip install ultralytics 2>/dev/null || python3 -m pip install ultralytics

echo ""
echo "=== Step 3: PyTorch → ONNX ==="
if [ ! -f yolov8n.onnx ]; then
    python3 -c "
from ultralytics import YOLO
model = YOLO('models/yolov8n.pt')
model.export(format='onnx', opset=17, imgsz=640, dynamic=False, simplify=True)
print('✓ yolov8n.onnx 导出完成')
"
    mv yolov8n.onnx "$MODEL_DIR/" 2>/dev/null || true
else
    echo "  ✓ yolov8n.onnx 已存在, 跳过"
fi

echo ""
echo "=== Step 4: ONNX → TensorRT Engine (FP16) ==="
if [ ! -f yolov8n.engine ]; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$MODEL_DIR/yolov8n.onnx" \
        --saveEngine="$MODEL_DIR/yolov8n.engine" \
        --fp16 \
        --workspace=2048 \
        --optShapes=input:6x3x640x640 \
        --minShapes=input:1x3x640x640 \
        --maxShapes=input:6x3x640x640
    echo "  ✓ yolov8n.engine 构建完成"
else
    echo "  ✓ yolov8n.engine 已存在, 跳过"
fi

echo ""
echo "=== Step 5: 下载 COCO labels ==="
if [ ! -f labels.txt ]; then
    python3 -c "
labels = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]
with open('models/labels.txt', 'w') as f:
    f.write('\n'.join(labels))
print(f'✓ labels.txt 写入 {len(labels)} 个类别')
"
else
    echo "  ✓ labels.txt 已存在, 跳过"
fi

echo ""
echo "=== Step 6: 验证产物 ==="
echo ""
for f in yolov8n.pt yolov8n.onnx yolov8n.engine labels.txt; do
    if [ -f "$MODEL_DIR/$f" ]; then
        SIZE=$(du -h "$MODEL_DIR/$f" | cut -f1)
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ $f 缺失"
    fi
done

echo ""
echo "=== 完成 ==="
echo "产物路径: $MODEL_DIR"
echo "下一步: 修改 configs/app/app.yaml 中的 model_engine_path 指向 models/yolov8n.engine"