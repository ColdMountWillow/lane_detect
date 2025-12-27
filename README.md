# 车道线检测实验

基于霍夫变换的车道线检测系统，使用 Python + OpenCV 实现。

## 环境依赖

- Python 3.6+
- opencv-python
- numpy

## 安装方式

```bash
pip install opencv-python numpy
```

## 使用方法

### 基本用法

处理单张图像：

```bash
python lane_detect.py --input data/test.jpg --output outputs
```

处理目录中的所有图像：

```bash
python lane_detect.py --input data/ --output outputs
```

显示处理结果窗口：

```bash
python lane_detect.py --input data/test.jpg --output outputs --show
```

### 参数说明

#### 必需参数

- `--input` / `-i`: 输入图像路径或目录路径

#### 可选参数

- `--output` / `-o`: 输出目录（默认: `outputs`）
- `--show` / `-s`: 显示处理结果窗口

#### 算法参数

**Canny 边缘检测参数：**

- `--canny1`: Canny 低阈值（默认: 50）
- `--canny2`: Canny 高阈值（默认: 150）

**霍夫变换参数：**

- `--hough_thresh`: HoughLinesP 累加器阈值（默认: 50）
- `--min_line_len`: 线段最小长度（默认: 50）
- `--max_line_gap`: 线段间最大间隙（默认: 100）

**ROI 参数：**

- `--roi_top_ratio`: ROI 顶部高度比例，0-1 之间（默认: 0.6）
  - 0.6 表示从图像高度的 60% 处开始保留道路区域

**斜率筛选参数：**

- `--slope_thresh`: 斜率阈值，过滤接近水平的线（默认: 0.5）

### 参数调优示例

如果检测到的线段太多（噪声多）：

```bash
python lane_detect.py --input data/test.jpg --hough_thresh 80 --min_line_len 80
```

如果检测到的线段太少（漏检）：

```bash
python lane_detect.py --input data/test.jpg --hough_thresh 30 --min_line_len 30 --max_line_gap 150
```

如果边缘检测效果不好：

```bash
# 增强边缘检测（降低阈值）
python lane_detect.py --input data/test.jpg --canny1 30 --canny2 100

# 减弱边缘检测（提高阈值）
python lane_detect.py --input data/test.jpg --canny1 80 --canny2 200
```

## 输出说明

- 输出目录：默认 `outputs/`，可通过 `--output` 指定
- 输出文件名：原文件名 + `_lane.png` 后缀
  - 例如：`test.jpg` → `test_lane.png`
- 输出内容：在原图上绘制检测到的左右车道线（绿色）

## 处理流程

1. 读取图像
2. 灰度化
3. 高斯滤波降噪
4. Canny 边缘检测
5. ROI 遮罩（保留下半部分道路区域）
6. 霍夫变换检测直线
7. 斜率筛选（分为左右车道）
8. 线段聚合拟合
9. 绘制并保存结果

## 注意事项

- 输入图像应包含清晰的车道线
- 建议使用道路视角的图像（类似行车记录仪视角）
- 如果检测效果不佳，可尝试调整参数
- 程序会自动创建输出目录（如果不存在）
