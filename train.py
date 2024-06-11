import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

# 注册训练集和验证集
register_coco_instances("mtwi_train", {}, os.path.join(label_folder, "detectron2_format.json"), image_folder)
register_coco_instances("mtwi_test", {}, "path/to/test/json", "path/to/test/image")  # 如果有测试集的话

# 获取数据集的元数据
metadata = MetadataCatalog.get("mtwi_train")

# 创建配置
cfg = get_cfg()
cfg.merge_from_file("path/to/config.yaml")  # 使用预定义的配置文件
cfg.DATASETS.TRAIN = ("mtwi_train",)  # 训练集
cfg.DATASETS.TEST = ()                 # 无验证集
cfg.DATALOADER.NUM_WORKERS = 4         # 数据加载器的工作线程数
cfg.MODEL.WEIGHTS = "path/to/pretrained/model.pth"  # 使用预训练的模型
cfg.SOLVER.IMS_PER_BATCH = 2           # batch size
cfg.SOLVER.BASE_LR = 0.001             # 学习率
cfg.SOLVER.MAX_ITER = 1000             # 最大迭代次数
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # 训练时的RoI批大小
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # 类别数

# 创建输出目录
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 创建并运行训练器
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
