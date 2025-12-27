#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车道线检测 - 基于霍夫变换
使用 Canny 边缘检测 + ROI + HoughLinesP 检测车道线
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def gaussian_blur(img, kernel_size=5):
    """
    高斯滤波降噪
    
    Args:
        img: 输入图像
        kernel_size: 高斯核大小（必须是奇数）
    
    Returns:
        滤波后的图像
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """
    Canny 边缘检测
    
    Args:
        img: 输入灰度图像
        low_threshold: 低阈值
        high_threshold: 高阈值
    
    Returns:
        边缘检测结果（二值图）
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, top_ratio=0.55):
    """
    创建 ROI 遮罩，只保留下半部分道路区域
    
    Args:
        img: 输入图像
        top_ratio: ROI 顶部高度比例（0-1），0.55 表示从图像高度的 55% 处开始保留
    
    Returns:
        遮罩后的图像
    """
    height, width = img.shape
    # 创建全零遮罩
    mask = np.zeros_like(img)
    # 定义多边形顶点（梯形，保留下半部分）
    # 底部两个角点：左下角和右下角
    # 顶部两个角点：稍微向内收缩，形成梯形（符合透视效果）
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)
    top_left = (int(width * 0.05), int(height * top_ratio))
    top_right = (int(width * 0.95), int(height * top_ratio))
    
    # 填充多边形区域为白色（255）
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    # 应用遮罩
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def hough_lines_detection(edges, threshold=50, min_line_len=50, max_line_gap=100):
    """
    使用概率霍夫变换检测直线
    
    Args:
        edges: 边缘检测结果
        threshold: 累加器阈值（检测直线所需的最少交点）
        min_line_len: 线段最小长度
        max_line_gap: 线段间最大间隙（可连接成一条线）
    
    Returns:
        检测到的线段列表，每个元素为 [x1, y1, x2, y2]
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,              # 距离精度（像素）
        theta=np.pi/180,    # 角度精度（弧度）
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    return lines.tolist()


def filter_lane_lines(lines, slope_thresh=0.5, img_width=None):
    """
    通过斜率筛选车道线，分为左右车道
    
    Args:
        lines: 线段列表，每个元素为 [x1, y1, x2, y2]
        slope_thresh: 斜率阈值，小于此值的线（接近水平）将被过滤
        img_width: 图像宽度，用于更智能的左右分类
    
    Returns:
        left_lines: 左车道线候选（负斜率或位于左半部分）
        right_lines: 右车道线候选（正斜率或位于右半部分）
    """
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算斜率（注意图像坐标系 y 向下）
        if x2 - x1 == 0:  # 避免除零
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        
        # 过滤接近水平的线
        if abs(slope) < slope_thresh:
            continue
        
        # 负斜率：左车道线（从左上到右下）
        # 正斜率：右车道线（从左下到右上）
        # 如果斜率接近0但通过了阈值，根据位置判断
        if img_width is not None:
            line_center_x = (x1 + x2) / 2
            if slope < 0 or (abs(slope) < 1.0 and line_center_x < img_width / 2):
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
        else:
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
    
    return left_lines, right_lines


def fit_lane_line(lines, img_shape, roi_top_ratio=0.6, draw_top_ratio=0.35):
    """
    对多条线段进行聚合，拟合出一条代表车道线
    
    方法：收集所有线段的端点，使用 np.polyfit 拟合直线
    
    Args:
        lines: 线段列表，每个元素为 [x1, y1, x2, y2]
        img_shape: 图像形状 (height, width)
        roi_top_ratio: ROI 顶部高度比例（用于检测）
        draw_top_ratio: 绘制起点高度比例（用于绘制，通常比 ROI 更靠上）
    
    Returns:
        (x1, y1, x2, y2): 拟合后的线段端点，如果拟合失败返回 None
    """
    if len(lines) == 0:
        return None
    
    # 收集所有点
    x_points = []
    y_points = []
    
    for x1, y1, x2, y2 in lines:
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])
    
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    # 如果点太少，无法拟合
    if len(x_points) < 2:
        return None
    
    try:
        # 使用一次多项式（直线）拟合
        # 注意：使用 y 作为自变量，x 作为因变量，因为车道线接近垂直
        coeffs = np.polyfit(y_points, x_points, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # 计算直线在图像范围内的端点
        height, width = img_shape[:2]
        # 绘制起点：从图像上方开始（draw_top_ratio），而不是从 ROI 顶部开始
        # 这样可以绘制完整的车道线
        y1 = int(height * draw_top_ratio)  # 绘制起点（图像上方）
        y2 = height - 1                      # 图像底部
        
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        
        # 确保点在图像范围内
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        
        return (x1, y1, x2, y2)
    
    except Exception as e:
        return None


def draw_lane_lines(img, left_line, right_line, color=(0, 255, 0), thickness=8):
    """
    在原图上绘制车道线
    
    Args:
        img: 原始图像（BGR格式）
        left_line: 左车道线 (x1, y1, x2, y2) 或 None
        right_line: 右车道线 (x1, y1, x2, y2) 或 None
        color: 线条颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    result = img.copy()
    
    # 如果两条线都存在，可以绘制填充区域（可选）
    if left_line is not None and right_line is not None:
        x1_l, y1_l, x2_l, y2_l = left_line
        x1_r, y1_r, x2_r, y2_r = right_line
        
        # 创建填充区域（车道区域）
        lane_area = np.array([
            [x1_l, y1_l],
            [x2_l, y2_l],
            [x2_r, y2_r],
            [x1_r, y1_r]
        ], np.int32)
        
        # 绘制半透明填充（可选，注释掉以保持简洁）
        # overlay = result.copy()
        # cv2.fillPoly(overlay, [lane_area], (0, 255, 0))
        # cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    # 绘制左车道线
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制右车道线
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    return result


def process_single_image(img_path, output_dir, args):
    """
    处理单张图像
    
    Args:
        img_path: 输入图像路径
        output_dir: 输出目录
        args: 命令行参数
    
    Returns:
        success: 是否成功处理
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return False
    
    print(f"\n处理图像: {img_path}")
    
    # 1. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯滤波降噪
    blurred = gaussian_blur(gray, kernel_size=5)
    
    # 3. Canny 边缘检测
    edges = canny_edge_detection(blurred, args.canny1, args.canny2)
    
    # 4. ROI 遮罩
    roi_edges = region_of_interest(edges, args.roi_top_ratio)
    
    # 5. 霍夫变换检测直线
    lines = hough_lines_detection(
        roi_edges,
        threshold=args.hough_thresh,
        min_line_len=args.min_line_len,
        max_line_gap=args.max_line_gap
    )
    
    print(f"  检测到线段数量: {len(lines)}")
    
    # 6. 斜率筛选，分为左右车道
    left_lines, right_lines = filter_lane_lines(lines, args.slope_thresh, img.shape[1])
    print(f"  左车道候选数量: {len(left_lines)}")
    print(f"  右车道候选数量: {len(right_lines)}")
    
    # 7. 拟合左右车道线
    # 绘制起点设置为图像高度的指定比例，值越小线条越长（从更上方开始）
    draw_top_ratio = args.draw_top_ratio if hasattr(args, 'draw_top_ratio') else 0.25
    left_line = fit_lane_line(left_lines, img.shape, args.roi_top_ratio, draw_top_ratio) if left_lines else None
    right_line = fit_lane_line(right_lines, img.shape, args.roi_top_ratio, draw_top_ratio) if right_lines else None
    
    if left_line is not None:
        print(f"  ✓ 成功拟合左车道线")
    else:
        print(f"  ✗ 未能拟合左车道线")
    
    if right_line is not None:
        print(f"  ✓ 成功拟合右车道线")
    else:
        print(f"  ✗ 未能拟合右车道线")
    
    # 8. 绘制车道线
    result = draw_lane_lines(img, left_line, right_line)
    
    # 9. 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name = Path(img_path).stem
    output_path = output_dir / f"{img_name}_lane.png"
    cv2.imwrite(str(output_path), result)
    print(f"  结果已保存: {output_path}")
    
    # 10. 可选显示
    if args.show:
        cv2.imshow("原始图像", img)
        cv2.imshow("边缘检测", edges)
        cv2.imshow("ROI 边缘", roi_edges)
        cv2.imshow("车道线检测结果", result)
        print("  按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="车道线检测 - 基于霍夫变换",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入图像路径或目录路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="输出目录（默认: outputs）"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="显示处理结果窗口"
    )
    
    # Canny 参数
    parser.add_argument(
        "--canny1",
        type=int,
        default=50,
        help="Canny 低阈值（默认: 50）"
    )
    
    parser.add_argument(
        "--canny2",
        type=int,
        default=150,
        help="Canny 高阈值（默认: 150，建议保持与低阈值比例约1:3）"
    )
    
    # Hough 参数
    parser.add_argument(
        "--hough_thresh",
        type=int,
        default=40,
        help="HoughLinesP 累加器阈值（默认: 40，降低以检测更多线段）"
    )
    
    parser.add_argument(
        "--min_line_len",
        type=int,
        default=30,
        help="线段最小长度（默认: 30，降低以检测虚线车道线）"
    )
    
    parser.add_argument(
        "--max_line_gap",
        type=int,
        default=150,
        help="线段间最大间隙（默认: 150，用于连接虚线车道线）"
    )
    
    # ROI 参数
    parser.add_argument(
        "--roi_top_ratio",
        type=float,
        default=0.55,
        help="ROI 顶部高度比例（0-1，默认: 0.55，保留更多道路区域）"
    )
    
    # 斜率阈值
    parser.add_argument(
        "--slope_thresh",
        type=float,
        default=0.3,
        help="斜率阈值，过滤接近水平的线（默认: 0.3，降低以保留更多候选）"
    )
    
    parser.add_argument(
        "--draw_top_ratio",
        type=float,
        default=0.3,
        help="绘制起点高度比例（0-1，默认: 0.3，值越小线条越长，从更上方开始绘制）"
    )
    
    args = parser.parse_args()
    
    # 处理输入路径
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"错误：输入路径不存在: {args.input}")
        return
    
    # 判断是文件还是目录
    if input_path.is_file():
        # 单张图像
        image_files = [input_path]
    else:
        # 目录，查找所有图像文件
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        ]
        
        if len(image_files) == 0:
            print(f"错误：目录中没有找到图像文件: {args.input}")
            return
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"参数设置:")
    print(f"  Canny: [{args.canny1}, {args.canny2}]")
    print(f"  Hough: threshold={args.hough_thresh}, min_len={args.min_line_len}, max_gap={args.max_line_gap}")
    print(f"  ROI top ratio: {args.roi_top_ratio}")
    print(f"  Slope threshold: {args.slope_thresh}")
    
    # 处理每张图像
    success_count = 0
    for img_file in image_files:
        if process_single_image(img_file, args.output, args):
            success_count += 1
    
    print(f"\n处理完成: {success_count}/{len(image_files)} 张图像成功")


if __name__ == "__main__":
    main()

