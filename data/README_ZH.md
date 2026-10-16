# 数据说明 (Data Description)

[English](./README.md)

本目录包含心脑血管疾病大模型评测基准所使用的各类数据。

## 数据下载

评测所需的数据集已发布在 OpenI 启智社区，请通过以下链接下载：

- **下载链接**：[CardioVascBench 数据集](https://openi.pcl.ac.cn/explore/datasets/cindyc/CardioVascBench)

下载完成后，请按照以下目录结构将数据解压或放置到相应位置。

## 目录结构

- `source_data/`: 包含原始生成的病例数据（如 `source_data/source_data_public.json`）。
- `llm_eval_data/`: 包含经过清洗和标准化的评估集数据（如 `eval_data/public_diagnosis_eval.json`）。
- `llms_answers_data/`: 用于存放各个待测大模型生成的诊断回答。
- `llms_answers_data_generator.py`: 用于调用模型 API 并生成回答数据的脚本。

---
有关评测流程的更多信息，请参阅根目录下的 [README_ZH.md](../README_ZH.md)。
