# yolov11_d435i_detection
**使用realsense d435i相机，基于pytorch实现yolov11目标检测，实时返回检测目标相机坐标系下的位置信息。**

# 1.Environment：

1.一个可以运行YOLOv11的python环境

1.1
```bash
# 创建一个 Python 3.10 的环境（环境名为 yolov11）
conda create -n yolov11 python=3.10 -y
```
```bash
# 激活新创建的环境
conda activate yolov11
```

1.2
```bash
# 安装 PyTorch（含 CUDA ）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

1.3
```bash
# 安装项目依赖
pip install -r requirements.txt
```

2.一个realsense相机和pyrealsense2库

```bash
pip install pyrealsense2
```

**在以下环境中测试成功**

- **ubuntu 20.04.1** python 3.10 Pytorch 2.4.1+gpu CUDA 12.4 NVIDIA GeForce RTX 4060 Laptop GPU

# 2.Results：

- Colorimage:


# 3.Model config：

修改模型配置文件，这里以yolov11n模型为例。也可以使用自己训练的权重模型。

```yaml
weight:  "weights/yolov11n.pt"
# 输入图像的尺寸
input_size: 640
# 类别个数
class_num:  80
# 标签名称
class_name: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
# 阈值设置
threshold:
  iou: 0.45
  confidence: 0.6
# 计算设备
# - cpu
# - 0 <- 使用GPU
device: '0'
```

# 4.Camera config：

分辨率只能改特定的参数，不然会报错。d435i可以用 1280x720, 640x480, 848x480。

```python
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
```
# 5.code return xyz：
下方代码实现从像素坐标系到相机坐标系转换，并且标注中心点以及三维坐标信息。
```python
# 深度坐标投影
camera_xyz_list = []
for cid, xyxy in zip(cls_ids, boxes):
    # 1. 计算检测框中心像素 (u, v)
    x1, y1, x2, y2 = map(int, xyxy)
    ux, uy = (x1 + x2) // 2, (y1 + y2) // 2

    # 2. 读取该像素的深度值（以米为单位）
    dist = depth_frame.get_distance(ux, uy)

    # 3. 像素坐标 + 深度 → 相机坐标
    xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dist)
    xyz = np.round(xyz, 3).tolist()
    camera_xyz_list.append(xyz)

    # 4. 在图像上画出中心点并标注三维坐标
    cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 2)
    cv2.putText(canvas, str(xyz), (ux + 10, uy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
```
# 6.Reference:

[https://github.com/ultralytics](https://github.com/ultralytics)

[https://github.com/Thinkin99/yolov5_d435i_detection](https://github.com/Thinkin99/yolov5_d435i_detection)
