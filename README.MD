## ICCV 2021 Meituan Large Fine Food AI Workshop

---
### **队伍简介**

队名：GradientBoostingTeam，队长：鱼丸粗面，邮箱：zhuoyin94@163.com。20210614

---
### **源码说明**

基础源码文件说明：
- **input_pipeline.py**：模型训练程序的入口。用于构造对于模型的训练数据高效输入的数据管道，其中集成各种数据增强策略。
- **models.py**：基础模型库，待添加：ResNet，ResNext，BiT等。
- **logger.py**：训练日志文件记录，待完善。
- **dingtalk_remote_monitor.py**：用于远程模型训练监控的callback，每一个epoch向远程dingtalk发送meta-info。后续将与logger模块合并。
- **visualizing_data.py**：原始数据基础可视化。

我们采用了[该项目](https://github.com/MichaelYin1994/python-style-guide)中的编码规范，该规范大部分脱胎于Google编码规范，力求使得代码简洁易读。

---
### **文件结构**

```
.
├── cached_data
├── ckpt
├── data
├── dingtalk_remote_monitor.py
├── input_pipeline.py
├── logger.py
├── models.py
├── README.MD
├── submissions
└── visualizing_data.py
```

文件结构说明：
- **./data/**：用于存储原始数据，并切分了部分原始数据便于调试。
- **./ckpt/**：模型训练权重checkpoint存储位置。
- **./cached_data/**：用于存储预处理后的临时数据，后续用于存储tfrecord形式的数据，加速IO过程。
- **./submissions/**：用于存储模型预测的测试集待提交的submissions结果。
