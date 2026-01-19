# 数据准备脚本使用说明

本文档介绍如何使用数据准备脚本将各种心音数据集转换为 PA-HCL 预处理流程所需的格式。

---

## 目录

1. [概述](#概述)
2. [支持的数据集](#支持的数据集)
3. [快速开始](#快速开始)
4. [数据集准备详细指南](#数据集准备详细指南)
5. [脚本使用说明](#脚本使用说明)
6. [故障排查](#故障排查)

---

## 概述

PA-HCL 预处理流程要求数据按**受试者（subject）**组织，每个受试者的录音放在独立的文件夹中：

```
data/raw/
├── subject_0001/
│   ├── rec_01.wav
│   ├── rec_02.wav
│   └── rec_03.wav
├── subject_0002/
│   ├── rec_01.wav
│   └── rec_02.wav
└── ...
```

数据准备脚本会自动将各种公开数据集和自建数据集转换为这种格式。

---

## 支持的数据集

### 1. CirCor DigiScope Dataset
- **来源**: https://physionet.org/content/circor-heart-sound/
- **特点**: 多位置听诊记录（AV, MV, PV, TV），包含诊断标签
- **规模**: ~1000 名受试者，~5000 个录音
- **下载**: 自动下载脚本支持

### 2. PhysioNet 2016 Challenge Dataset
- **来源**: https://physionet.org/content/challenge-2016/
- **特点**: 标准化的心音分类数据集（正常/异常）
- **规模**: ~3000 个录音
- **下载**: 自动下载脚本支持

### 3. Pascal Challenge Dataset
- **来源**: https://istethoscope.peterjbentley.com/heartchallenge/
- **特点**: 多种心脏疾病类别
- **规模**: ~1000 个录音
- **下载**: 需要手动下载

### 4. 自建数据集
- **格式**: 按疾病类别（Normal/Abnormal）组织
- **文件命名**: `<condition>_case<id>_<metadata>_<location>.wav`
- **支持**: 自动解析文件名提取元数据

---

## 快速开始

### 方法 1: 使用快速启动脚本（推荐）

```bash
# 1. 检查环境和依赖
./quickstart.sh check

# 2. 准备单个数据集（完整流程：下载→准备→预处理）
./quickstart.sh full circor

# 或者分步执行
./quickstart.sh download circor      # 下载
./quickstart.sh prepare circor       # 准备（转换格式）
./quickstart.sh preprocess circor    # 预处理（切分周期）

# 3. 准备自建数据集
./quickstart.sh prepare custom
./quickstart.sh preprocess custom
```

### 方法 2: 手动执行各个脚本

```bash
# 1. 安装依赖
pip install pandas tqdm requests openpyxl

# 2. 下载公开数据集
python download_datasets.py --dataset circor --output-dir ./data/downloads

# 3. 准备数据集
python prepare_circor.py \
    --input-dir ./data/downloads/extracted/the-circor-digiscope-phonocardiogram-dataset-1.0.3 \
    --output-dir ./data/raw/circor

# 4. 运行预处理
cd ../..
python scripts/preprocess.py --raw_dir data/raw/circor --output_dir data/processed
```

---

## 数据集准备详细指南

### CirCor DigiScope Dataset

#### 下载方式

**自动下载**：
```bash
python download_datasets.py --dataset circor --output-dir ./data/downloads
```

**手动下载**：
1. 访问 https://physionet.org/content/circor-heart-sound/
2. 注册 PhysioNet 账户并登录
3. 下载 `circor-heart-sound-1.0.3.zip`
4. 解压到 `./data/downloads/extracted/`

#### 数据准备

```bash
python prepare_circor.py \
    --input-dir ./data/downloads/extracted/the-circor-digiscope-phonocardiogram-dataset-1.0.3 \
    --output-dir ./data/raw/circor \
    --verbose
```

**参数说明**：
- `--input-dir`: 解压后的数据集目录
- `--output-dir`: 输出目录（按受试者组织）
- `--copy`: 复制文件（默认创建符号链接以节省空间）
- `--verbose`: 显示详细日志

**输出结构**：
```
data/raw/circor/
├── subject_10001/
│   ├── rec_AV.wav  # Aortic Valve
│   ├── rec_MV.wav  # Mitral Valve
│   ├── rec_PV.wav  # Pulmonary Valve
│   └── rec_TV.wav  # Tricuspid Valve
├── subject_10002/
│   └── ...
└── circor_metadata.csv
```

---

### PhysioNet 2016 Challenge Dataset

#### 下载方式

**自动下载**：
```bash
python download_datasets.py --dataset physionet2016 --output-dir ./data/downloads
```

#### 数据准备

```bash
python prepare_physionet2016.py \
    --input-dir ./data/downloads/extracted/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0 \
    --output-dir ./data/raw/physionet2016 \
    --include-validation \
    --verbose
```

**参数说明**：
- `--include-validation`: 包含验证集（默认只处理训练集）
- 其他参数同 CirCor

**输出结构**：
```
data/raw/physionet2016/
├── subject_a0001/
│   └── rec_01.wav
├── subject_a0002/
│   └── rec_01.wav
└── physionet2016_metadata.csv
```

---

### Pascal Challenge Dataset

#### 下载方式

**需要手动下载**：
1. 访问 https://istethoscope.peterjbentley.com/heartchallenge/
2. 阅读并接受使用条款
3. 下载数据集
4. 解压到 `./data/downloads/extracted/heartchallenge/`

#### 数据准备

```bash
python prepare_pascal.py \
    --input-dir ./data/downloads/extracted/heartchallenge \
    --output-dir ./data/raw/pascal \
    --sets set_a set_b \
    --verbose
```

**参数说明**：
- `--sets`: 要处理的子集（默认: set_a set_b）

**输出结构**：
```
data/raw/pascal/
├── subject_a0001/
│   └── rec_01.wav
├── subject_b0001/
│   └── rec_01.wav
└── pascal_labels.csv
```

---

### 自建数据集

#### 输入数据格式要求

您的自建数据集应按以下结构组织：

```
data/raw/
├── Abnormal/
│   ├── asd_case0001_female_4_20s_USA_A.wav
│   ├── asd_case0001_female_4_20s_USA_E.wav
│   ├── asd_case0001_female_4_20s_USA_M.wav
│   ├── asd_case0001_female_4_20s_USA_P.wav
│   ├── asd_case0001_female_4_20s_USA_T.wav
│   └── ...
├── Normal/
│   ├── normal_case0001_male_6_20s_USA_A.wav
│   ├── normal_case0001_male_6_20s_USA_E.wav
│   └── ...
└── metadata.xlsx (可选)
```

#### 文件命名规则

标准格式：`<condition>_case<id>_<gender>_<age>_<duration>_<country>_<location>.wav`

**示例**：
- `asd_case0001_female_4_20s_USA_A.wav`
  - condition: asd (房间隔缺损)
  - case_id: 0001
  - gender: female
  - age: 4
  - duration: 20s
  - country: USA
  - location: A (Aortic)

**位置代码**：
- `A`: Aortic (主动脉瓣区)
- `E`: Erb's point (Erb 点)
- `M`: Mitral (二尖瓣区)
- `P`: Pulmonary (肺动脉瓣区)
- `T`: Tricuspid (三尖瓣区)

#### 数据准备

```bash
python prepare_custom.py \
    --input-dir ./data/raw \
    --output-dir ./data/raw/custom_organized \
    --verbose
```

**输出结构**：
```
data/raw/custom_organized/
├── subject_asd_case0001/
│   ├── rec_A.wav
│   ├── rec_E.wav
│   ├── rec_M.wav
│   ├── rec_P.wav
│   └── rec_T.wav
├── subject_normal_case0001/
│   ├── rec_A.wav
│   └── ...
├── custom_metadata.csv      # 自动生成的元数据
└── original_metadata.csv    # 原始元数据（如果提供了 metadata.xlsx）
```

**生成的元数据内容**：
```csv
subject_id,case_id,condition,location,gender,age,duration,country,original_filename,organized_path
subject_asd_case0001,0001,asd,A,female,4,20s,USA,asd_case0001_female_4_20s_USA_A.wav,subject_asd_case0001/rec_A.wav
```

---

## 脚本使用说明

### download_datasets.py

自动下载公开数据集。

```bash
# 下载所有数据集
python download_datasets.py --output-dir ./data/downloads

# 下载特定数据集
python download_datasets.py --dataset circor --output-dir ./data/downloads

# 只下载不解压
python download_datasets.py --dataset circor --no-extract

# 只解压已下载的文件
python download_datasets.py --dataset circor --extract-only

# 保留压缩包
python download_datasets.py --dataset circor --keep-archive
```

### prepare_data.py (统一入口)

统一的数据准备入口，可以自动检测数据集类型。

```bash
# 自动检测数据集类型
python prepare_data.py --input-dir /path/to/dataset --output-dir ./data/raw/organized

# 手动指定数据集类型
python prepare_data.py --dataset-type circor --input-dir /path/to/circor

# 准备所有数据集（从基础目录）
python prepare_data.py --dataset-type all --base-dir /path/to/datasets
```

### quickstart.sh

快速启动脚本，提供一键式工作流。

```bash
# 检查环境
./quickstart.sh check

# 下载数据集
./quickstart.sh download circor
./quickstart.sh download physionet2016
./quickstart.sh download all  # 下载所有

# 准备数据集
./quickstart.sh prepare circor
./quickstart.sh prepare custom

# 运行预处理
./quickstart.sh preprocess circor

# 完整流程（下载→准备→预处理）
./quickstart.sh full circor
```

---

## 常见问题

### 1. 符号链接 vs 文件复制

**默认行为**：创建符号链接（symlink）
- **优点**：节省磁盘空间，不复制原始文件
- **缺点**：如果移动或删除原始文件，链接会失效

**使用 `--copy` 参数**：复制文件
- **优点**：独立的副本，不依赖原始文件
- **缺点**：占用更多磁盘空间

```bash
# 创建符号链接（推荐）
python prepare_circor.py --input-dir /path/to/data --output-dir ./data/raw/circor

# 复制文件
python prepare_circor.py --input-dir /path/to/data --output-dir ./data/raw/circor --copy
```

### 2. 如何验证准备结果

```bash
# 检查输出目录结构
tree -L 2 data/raw/circor

# 检查受试者数量
ls -1 data/raw/circor | wc -l

# 检查录音文件数量
find data/raw/circor -name "*.wav" | wc -l

# 查看元数据
head -n 20 data/raw/circor/circor_metadata.csv
```

### 3. 文件命名不规范怎么办？

对于自建数据集，如果文件命名不符合标准格式，脚本会尝试智能解析：
- 日志会显示哪些文件解析失败
- 可以手动修改文件名后重新运行
- 或者修改 `parse_filename()` 函数以适应您的命名规则

### 4. 下载速度慢或失败

```bash
# 方法 1: 使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
python download_datasets.py --dataset circor

# 方法 2: 手动下载后解压
python download_datasets.py --dataset circor --extract-only

# 方法 3: 使用 wget/curl 手动下载
wget https://physionet.org/static/published-projects/circor-heart-sound/...
python download_datasets.py --dataset circor --extract-only
```

### 5. 预处理失败

如果数据准备后预处理失败，检查：

1. **文件格式**：确保是有效的 WAV 文件
```bash
file data/raw/circor/subject_10001/rec_AV.wav
```

2. **采样率**：查看采样率是否正常
```bash
ffprobe data/raw/circor/subject_10001/rec_AV.wav 2>&1 | grep "Audio"
```

3. **文件完整性**：确保文件没有损坏
```bash
python -c "
import scipy.io.wavfile as wavfile
sr, data = wavfile.read('data/raw/circor/subject_10001/rec_AV.wav')
print(f'Sample rate: {sr}, Duration: {len(data)/sr:.2f}s')
"
```

---

## 下一步

数据准备完成后，进行预处理：

```bash
# 预处理单个数据集
python scripts/preprocess.py \
    --raw_dir data/raw/circor \
    --output_dir data/processed \
    --num_workers 4

# 使用配置文件
python scripts/preprocess.py \
    --config configs/default.yaml \
    --raw_dir data/raw/circor
```

预处理完成后，可以开始预训练：

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

---

## 脚本列表

| 脚本 | 功能 | 数据集 |
|------|------|--------|
| `download_datasets.py` | 自动下载公开数据集 | CirCor, PhysioNet 2016 |
| `prepare_circor.py` | 准备 CirCor 数据集 | CirCor |
| `prepare_physionet2016.py` | 准备 PhysioNet 2016 数据集 | PhysioNet 2016 |
| `prepare_pascal.py` | 准备 Pascal 数据集 | Pascal |
| `prepare_custom.py` | 准备自建数据集 | 自建数据 |
| `prepare_data.py` | 统一入口（自动检测） | 所有 |
| `quickstart.sh` | 快速启动脚本 | 所有 |

---

## 技术支持

如有问题，请：
1. 查看日志输出（使用 `--verbose` 参数）
2. 检查上述常见问题部分
3. 在项目 GitHub 提交 Issue
4. 联系项目维护者

---

**最后更新**: 2026-01-18
