# CAD Hatch Pattern Recognizer

用 AutoCAD `.pat` 合成数据训练 CNN，识别设计图纸中的填充图案名称。

## 支持的 10 种常用填充（含混凝土）

| 名称 | 说明 |
|------|------|
| ANSI31 | ANSI 铁、砖和石 |
| ANSI32 | ANSI 钢 |
| **AR-CONC** | **混凝土（随机点和石头）** |
| BRICK | 砖石表面 |
| STEEL | 钢材质 |
| EARTH | 地面 |
| LINE | 平行水平线 |
| NET | 水平/垂直栅格 |
| GRAVEL | 沙砾 |
| SOLID | 实体填充 |

当前模型测试集准确率约 **80%**。

## 快速开始

```bash
cd hatch_recognition
pip install -r requirements.txt

# 一键：生成数据 + 训练 + Web 界面
./run.sh

# 或分步：
python generate_dataset.py --samples 200 --size 128
python train.py --epochs 25 --batch-size 64
python predict.py path/to/hatch.png
python app.py   # http://localhost:7860
```

## 使用建议

上传时请尽量裁剪出**纯填充区域**（少文字、少标注线），效果更好。

## 目录

- `acad_4270.pat` — AutoCAD 填充定义
- `pat_parser.py` — PAT 解析（默认取上述 10 类）
- `renderer.py` — 填充渲染（比例/旋转/形状/干扰）
- `generate_dataset.py` — 合成训练集
- `train.py` / `predict.py` / `app.py` — 训练、推理、Web 上传
- `models/best_model.pt` — 已训练权重
