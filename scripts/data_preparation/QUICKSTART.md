# PA-HCL 数据准备快速上手指南

本指南帮助您快速开始使用 PA-HCL 的数据准备工具。

---

## 🚀 一分钟快速开始

### 对于公开数据集（CirCor, PhysioNet 2016, Pascal）

```bash
# 进入数据准备目录
cd scripts/data_preparation

# 方式 1: 使用快速启动脚本（推荐）
./quickstart.sh full circor           # 自动完成：下载→准备→预处理

# 方式 2: 分步执行
./quickstart.sh check                 # 检查环境
./quickstart.sh download circor       # 下载数据集
./quickstart.sh prepare circor        # 准备数据集
./quickstart.sh preprocess circor     # 运行预处理
```

### 对于自建数据集

```bash
# 1. 确保您的数据按以下结构组织：
# data/raw/
# ├── Normal/
# │   └── *.wav
# └── Abnormal/
#     └── *.wav

# 2. 运行准备脚本
cd scripts/data_preparation
./quickstart.sh prepare custom
./quickstart.sh preprocess custom
```

---

## 📋 各数据集详细步骤

### CirCor DigiScope Dataset

```bash
# 下载（约 10 GB）
python download_datasets.py --dataset circor --output-dir ./data/downloads

# 准备（转换为受试者格式）
python prepare_circor.py \
    --input-dir ./data/downloads/extracted/the-circor-digiscope-phonocardiogram-dataset-1.0.3 \
    --output-dir ./data/raw/circor

# 预处理（切分心动周期）
cd ../..
python scripts/preprocess.py --raw_dir data/raw/circor --output_dir data/processed
```

**期望输出**：
```
data/raw/circor/
├── subject_10001/
│   ├── rec_AV.wav
│   ├── rec_MV.wav
│   ├── rec_PV.wav
│   └── rec_TV.wav
└── ...
```

---

### PhysioNet 2016 Challenge Dataset

```bash
# 下载（约 1.2 GB）
python download_datasets.py --dataset physionet2016 --output-dir ./data/downloads

# 准备
python prepare_physionet2016.py \
    --input-dir ./data/downloads/extracted/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0 \
    --output-dir ./data/raw/physionet2016 \
    --include-validation

# 预处理
cd ../..
python scripts/preprocess.py --raw_dir data/raw/physionet2016 --output_dir data/processed
```

**期望输出**：
```
data/raw/physionet2016/
├── subject_a0001/
│   └── rec_01.wav
├── subject_a0002/
│   └── rec_01.wav
└── ...
```

---

### Pascal Challenge Dataset

```bash
# 1. 手动下载（需要注册）
# 访问: https://istethoscope.peterjbentley.com/heartchallenge/
# 下载后解压到: ./data/downloads/extracted/heartchallenge/

# 2. 准备
python prepare_pascal.py \
    --input-dir ./data/downloads/extracted/heartchallenge \
    --output-dir ./data/raw/pascal

# 3. 预处理
cd ../..
python scripts/preprocess.py --raw_dir data/raw/pascal --output_dir data/processed
```

---

### 自建数据集

**步骤 1: 组织您的原始数据**

按照以下结构组织：
```
data/raw/
├── Abnormal/
│   ├── asd_case0001_female_4_20s_USA_A.wav
│   ├── asd_case0001_female_4_20s_USA_E.wav
│   ├── asd_case0001_female_4_20s_USA_M.wav
│   └── ...
├── Normal/
│   ├── normal_case0001_male_6_20s_USA_A.wav
│   ├── normal_case0001_male_6_20s_USA_E.wav
│   └── ...
└── metadata.xlsx (可选)
```

**文件命名规则**：
```
<condition>_case<id>_<gender>_<age>_<duration>_<country>_<location>.wav
```

**位置代码**：
- `A`: Aortic (主动脉瓣区)
- `E`: Erb's point
- `M`: Mitral (二尖瓣区)
- `P`: Pulmonary (肺动脉瓣区)
- `T`: Tricuspid (三尖瓣区)

**步骤 2: 运行准备脚本**

```bash
python prepare_custom.py \
    --input-dir ./data/raw \
    --output-dir ./data/raw/custom_organized \
    --verbose
```

**步骤 3: 预处理**

```bash
cd ../..
python scripts/preprocess.py \
    --raw_dir data/raw/custom_organized \
    --output_dir data/processed
```

**期望输出**：
```
data/raw/custom_organized/
├── subject_asd_case0001/
│   ├── rec_A.wav
│   ├── rec_E.wav
│   └── rec_M.wav
├── subject_normal_case0001/
│   └── ...
└── custom_metadata.csv
```

---

## 🔍 验证结果

### 检查准备后的数据

```bash
# 查看目录结构
tree -L 2 data/raw/circor

# 统计受试者数量
ls -1d data/raw/circor/subject_* | wc -l

# 统计录音文件数量
find data/raw/circor -name "*.wav" | wc -l

# 查看元数据
head -n 10 data/raw/circor/circor_metadata.csv
```

### 检查预处理后的数据

```bash
# 查看处理后的数据结构
tree -L 2 data/processed

# 检查心动周期数量
find data/processed -name "*.npy" | wc -l

# 查看统计信息
cat data/processed/statistics.json | python -m json.tool
```

---

## ⚡ 常用命令速查

### 环境检查
```bash
./quickstart.sh check
```

### 下载所有公开数据集
```bash
python download_datasets.py --dataset all --output-dir ./data/downloads
```

### 一键准备并预处理
```bash
./quickstart.sh full circor          # CirCor
./quickstart.sh full physionet2016   # PhysioNet 2016
./quickstart.sh full custom          # 自建数据集
```

### 只准备不预处理
```bash
./quickstart.sh prepare circor
```

### 使用复制而非符号链接
```bash
python prepare_circor.py --input-dir /path/to/data --output-dir ./data/raw/circor --copy
```

---

## 🛠️ 故障排查

### 问题 1: 符号链接失效

**症状**：预处理时报错"文件不存在"

**解决**：使用 `--copy` 参数复制文件而非创建符号链接
```bash
python prepare_circor.py --input-dir /path/to/data --output-dir ./data/raw/circor --copy
```

### 问题 2: 文件命名不规范

**症状**：自建数据集准备时很多文件被跳过

**解决**：
1. 检查日志中的错误信息（使用 `--verbose`）
2. 确保文件命名符合规范
3. 或修改 `prepare_custom.py` 中的 `parse_filename()` 函数

### 问题 3: 预处理失败

**症状**：数据准备成功但预处理报错

**检查清单**：
```bash
# 1. 验证 WAV 文件格式
file data/raw/circor/subject_10001/rec_AV.wav

# 2. 检查采样率
ffprobe data/raw/circor/subject_10001/rec_AV.wav 2>&1 | grep Audio

# 3. 测试读取
python -c "
from scipy.io import wavfile
sr, data = wavfile.read('data/raw/circor/subject_10001/rec_AV.wav')
print(f'SR: {sr}, Duration: {len(data)/sr:.2f}s')
"
```

### 问题 4: 下载速度慢

**解决方案**：
```bash
# 使用代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port

# 或手动下载后解压
# 访问数据集网站手动下载
# 然后使用 --extract-only
python download_datasets.py --dataset circor --extract-only
```

---

## 📊 数据集规模参考

| 数据集 | 受试者数 | 录音数 | 总大小 | 预处理后 |
|--------|---------|--------|--------|----------|
| CirCor | ~1,000 | ~5,000 | ~10 GB | ~2 GB |
| PhysioNet 2016 | ~3,000 | ~3,000 | ~1.2 GB | ~500 MB |
| Pascal | ~1,000 | ~1,000 | ~500 MB | ~200 MB |

---

## 🔗 相关链接

- [完整使用文档](README.md)
- [项目使用文档](../../doc/项目使用文档.md)
- [预处理脚本说明](../preprocess.py)
- [预训练脚本说明](../pretrain.py)

---

## 💡 下一步

数据准备完成后：

1. **预训练模型**
   ```bash
   python scripts/pretrain.py --config configs/pretrain.yaml
   ```

2. **下游任务微调**
   ```bash
   python scripts/finetune.py --config configs/finetune.yaml
   ```

3. **模型评估**
   ```bash
   python scripts/evaluate.py --config configs/default.yaml
   ```

---

**祝您实验顺利！** 🎉
