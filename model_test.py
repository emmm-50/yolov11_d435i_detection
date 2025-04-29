import yaml

with open('config/yolov11s.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("类别数:", len(config['class_name']))
print("类别名:", config['class_name'])
