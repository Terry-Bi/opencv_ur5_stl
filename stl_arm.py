import cv2
import numpy as np

# 读取图像
image = cv2.imread('body.png')
if image is None:
    print("无法读取图像，请检查文件路径和完整性")
    exit()

# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 针对灰白物体优化的阈值处理
# 灰白物体在灰度图中亮度较高，使用较低的阈值下限来确保捕获
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# 可选：进行形态学操作，去除噪点并连接可能的断裂轮廓
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    print("未找到轮廓，请调整阈值参数")
else:
    # 取最大的轮廓（假设目标是最大的那个灰白物体）
    contour = max(contours, key=cv2.contourArea)
    # 计算最小外接矩形
    rect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), angle = rect
    # 确定长边和短边长度
    long_side = max(width, height)
    short_side = min(width, height)

    # 计算长边中线的两个端点
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # 计算长边的两个端点
    if width > height:
        long_start = box[0]
        long_end = box[2]
    else:
        long_start = box[1]
        long_end = box[3]
    # 长边中线的中点
    mid_point = ((long_start[0] + long_end[0]) // 2, (long_start[1] + long_end[1]) // 2)
    # 计算长边中线的方向向量
    vector = (long_end[0] - long_start[0], long_end[1] - long_start[1])
    # 计算垂直于长边中线的方向向量
    perpendicular_vector = (-vector[1], vector[0])
    # 归一化垂直方向向量
    norm = np.linalg.norm(perpendicular_vector)
    if norm != 0:
        perpendicular_vector = (perpendicular_vector[0] / norm, perpendicular_vector[1] / norm)
    # 计算垂线的两个端点
    short_half = short_side / 2
    short_start = (int(mid_point[0] - perpendicular_vector[0] * short_half), 
                  int(mid_point[1] - perpendicular_vector[1] * short_half))
    short_end = (int(mid_point[0] + perpendicular_vector[0] * short_half), 
                int(mid_point[1] + perpendicular_vector[1] * short_half))

    # 绘制长边中线和垂线
    cv2.line(image, tuple(long_start), tuple(long_end), (0, 0, 255), 2)
    cv2.line(image, short_start, short_end, (0, 255, 0), 2)

    # 输出坐标信息
    print("长边起点坐标:", long_start)
    print("长边终点坐标:", long_end)
    print("垂线起点坐标:", short_start)
    print("垂线终点坐标:", short_end)
    print("中点坐标:", mid_point)

    # 显示处理结果和阈值图像（便于调整参数）
    cv2.imshow('Threshold Image', thresh)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
