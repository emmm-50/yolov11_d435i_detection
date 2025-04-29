# -*- coding: utf-8 -*-
"""
完整的 YOLOv11 + Realsense 目标检测脚本
依赖：pyrealsense2, ultralytics, opencv-python, pyyaml, numpy
"""

import random
import yaml
import time
import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO


# ---------------------- Realsense 流程配置 ----------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)


def get_aligned_images():
    """
    等待并获取对齐后的彩色帧和深度帧，同时返回相机内参
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 相机内参
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    # 原始深度图与彩色图
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return intr, depth_intrin, color_image, depth_image, depth_frame


class Yolov11:
    def __init__(self, config_path='config/yolov11n.yaml'):
        # 载入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        # 加载 YOLOv11 模型
        self.model = YOLO(self.cfg['weight'])
        # 随机颜色
        self.colors = [
            [int(c) for c in np.random.randint(0, 255, size=3)]
            for _ in range(self.cfg['class_num'])
        ]

    def detect(self, img, canvas=None, view_img=True):
        """
        对一帧 BGR 图像执行检测，返回画框后的图和检测结果列表：
        canvas: 画好框的图
        class_ids: 类别索引列表
        boxes: 每个目标的 [x1,y1,x2,y2]
        confs: 置信度列表
        """
        results = self.model(img,verbose=False)  # 直接传入 numpy BGR 图像
        r = results[0]             # 仅处理第一张（批大小 1）
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        if view_img and canvas is None:
            canvas = img.copy()

        class_ids, coords, scores = [], [], []
        for xyxy, conf, cid in zip(boxes, confs, clss):
            class_ids.append(int(cid))
            coords.append(xyxy.tolist())
            scores.append(float(conf))
            if view_img:
                label = f"{self.cfg['class_name'][cid]} {conf:.2f}"
                self.plot_one_box(xyxy, canvas,
                                  color=self.colors[cid],
                                  label=label,
                                  line_thickness=3)
        return canvas, class_ids, coords, scores

    @staticmethod
    def plot_one_box(xyxy, img, color=(0, 255, 0),
                     label=None, line_thickness=2):
        """
        在 img 上绘制单个框和文字
        xyxy: [x1,y1,x2,y2]
        """
        x1, y1, x2, y2 = map(int, xyxy)
        tl = max(line_thickness, 1)
        # 矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
        # 标签
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0,
                        tl / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    print("[INFO] Yolov11目标检测-程序启动")
    yolov11 = Yolov11(config_path='config/yolov11n.yaml')
    print("[INFO] 完成YOLOv11模型加载")

    try:
        while True:
            intr, depth_intrin, color_img, depth_img, depth_frame = get_aligned_images()
            if color_img is None or not color_img.any():
                continue

            # 目标检测
            t0 = time.time()
            canvas, cls_ids, boxes, confs = yolov11.detect(color_img)
            t1 = time.time()

            # 深度坐标投影
            camera_xyz_list = []
            for cid, xyxy in zip(cls_ids, boxes):
                x1, y1, x2, y2 = map(int, xyxy)
                ux, uy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = depth_frame.get_distance(ux, uy)
                xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dist)
                xyz = np.round(xyz, 3).tolist()
                camera_xyz_list.append(xyz)
                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 2)
                cv2.putText(canvas, str(xyz), (ux + 10, uy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # FPS 显示
            fps = int(1.0 / (t1 - t0)) if t1 > t0 else 0
            cv2.putText(canvas, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # 展示
            cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
            cv2.imshow('detection', canvas)
            if cv2.waitKey(1) in (27, ord('q')):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
