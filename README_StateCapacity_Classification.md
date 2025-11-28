# 政府采购合同国家能力分类系统

## 基于 Berwick & Christia (2018) 《State Capacity Redux》理论框架

---

## 目录

1. [项目概述](#1-项目概述)
2. [理论框架](#2-理论框架)
3. [数据描述](#3-数据描述)
4. [分类方法论](#4-分类方法论)
5. [技术实现](#5-技术实现)
6. [使用指南](#6-使用指南)
7. [结果解读](#7-结果解读)
8. [局限性与展望](#8-局限性与展望)

---

## 1. 项目概述

### 1.1 研究背景

本项目旨在将中国政府采购合同数据按照国家能力理论框架进行分类，以便更好地理解政府采购行为与国家能力建设之间的关系。

### 1.2 数据来源

- **数据集**: 2012-2014年中国政府采购合同数据
- **记录数**: 848条合同记录
- **主要字段**: 合同名称、采购人、供应商、合同金额、所属行业等

### 1.3 分类目标

将每份政府采购合同分类为以下三种国家能力类型之一：
- **汲取能力 (Extractive Capacity)**
- **协调能力 (Coordination Capacity)**
- **合规能力 (Compliance Capacity)**

---

## 2. 理论框架

### 2.1 论文核心观点

Berwick & Christia (2018) 在《State Capacity Redux: Integrating Classical and Experimental Contributions to an Enduring Debate》中提出了一个整合经典与实验方法的国家能力分析框架。

### 2.2 三种国家能力定义

#### 2.2.1 汲取能力 (Extractive Capacity)

**定义**: 国家获取资源的能力，核心是统治者与资源持有者之间的关系。

**理论基础**:
- Levi (1988): 收入最大化假设
- Tilly (1990): 战争驱动财政发展的贝利西斯主义传统
- Scheve & Stasavage (2010): 国内冲突推动税收创新

**关键特征**:
- 税收和财政系统
- 资产和产权登记
- 资源管理和征收
- 财务审计和会计

**采购合同示例**:
- 税务系统升级项目
- 财务管理软件采购
- 资产评估服务
- 审计服务采购

#### 2.2.2 协调能力 (Coordination Capacity)

**定义**: 国家组织集体行动的能力，依赖官僚与社会成员的关系。

**理论基础**:
- Weber: 韦伯式专业官僚制
- Evans (1989): 发展型国家的"嵌入式自主"
- Johnson (1982): 日本通产省的行政指导卡特尔

**关键特征**:
- 基础设施建设
- 公共服务协调
- 行政办公系统
- 信息化和通讯网络

**采购合同示例**:
- 道路建设工程
- 办公设备采购
- 信息化系统建设
- 政府车辆采购

#### 2.2.3 合规能力 (Compliance Capacity)

**定义**: 确保公民、精英和官僚服从国家目标的能力。

**理论基础**:
- Bjorkman & Svensson (2009): 乌干达社区监督实验
- Reinikka & Svensson (2004): 教育服务提供
- Olken (2007, 2009): 腐败监控实验

**关键特征**:
- 公共服务提供（教育、医疗）
- 监管和执法
- 监督和监控机制
- 社会服务体系

**采购合同示例**:
- 学校教学设备
- 医院医疗器械
- 安防监控系统
- 培训服务采购

### 2.3 三种能力的相互关系

```
                    ┌─────────────┐
                    │   国家能力   │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  汲取能力   │ │  协调能力   │ │  合规能力   │
    │ Extractive  │ │Coordination │ │ Compliance  │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           │     资源 ──────┼───── 组织     │
           │               │               │
           └───────────────┼───────────────┘
                           │
                      相互依存
```

- **汲取能力 → 协调能力**: 没有资源，无法负担官僚机构
- **协调能力 → 汲取能力**: 协调能力降低汲取的交易成本
- **协调能力 → 合规能力**: 有效的协调促进公共服务提供
- **合规能力 → 汲取能力**: 合规确保税收的执行

---

## 3. 数据描述

### 3.1 数据集概览

| 年份 | 记录数 | 合同金额总计（万元） |
|------|--------|---------------------|
| 2012 | 51     | 2,401.99            |
| 2013 | 121    | 16,471.92           |
| 2014 | 676    | 7,061,228.84        |
| **总计** | **848** | **约70.8亿万元**  |

### 3.2 数据字段说明

| 字段分类 | 字段名称 | 说明 |
|----------|----------|------|
| 基本信息 | 年份、合同名称、合同编号 | 合同基础标识 |
| 交易主体 | 采购人、供应商 | 交易双方信息 |
| 地理信息 | 采购人_省/市/县、供应商_省/市/县 | 地理位置及行政区划 |
| 金额信息 | 合同金额、合同金额num_万元 | 交易金额 |
| 分类信息 | 所属地域、所属行业、采购方式 | 业务分类 |
| 内容描述 | 主要标的名称、规格型号 | 采购内容详情 |

### 3.3 数据分布特征

#### 地域分布（Top 5）
| 地域 | 合同数量 | 占比 |
|------|----------|------|
| 重庆市 | 535 | 63.1% |
| 云南省 | 24 | 2.8% |
| 甘肃省 | 21 | 2.5% |
| 北京市 | 17 | 2.0% |
| 安徽省 | 16 | 1.9% |

#### 行业分布（Top 5）
| 行业 | 合同数量 | 占比 |
|------|----------|------|
| 普通高等教育 | 87 | 10.3% |
| 综合医院 | 66 | 7.8% |
| 中等职业学校教育 | 63 | 7.4% |
| 普通小学教育 | 48 | 5.7% |
| 普通初中教育 | 40 | 4.7% |

---

## 4. 分类方法论

### 4.1 方法概述

由于缺乏人工标注的训练数据，本项目采用**半监督学习**方法：

1. **规则标注阶段**: 基于领域知识和关键词匹配生成伪标签
2. **机器学习阶段**: 使用伪标签训练多个分类模型
3. **模型选择阶段**: 通过交叉验证选择最优模型

### 4.2 规则标注设计

#### 4.2.1 关键词匹配规则

**汲取能力关键词**:
```python
extractive_keywords = [
    '税务', '税收', '财税', '财政', '预算', '审计', '会计',
    '财务', '资金', '收费', '资产', '资源', '矿产', '国土',
    '土地', '房产', '不动产', '登记', '产权', '确权', '征收'
]
```

**协调能力关键词**:
```python
coordination_keywords = [
    '道路', '公路', '桥梁', '交通', '水利', '电力', '通讯',
    '网络', '信息化', '建设', '工程', '施工', '办公', '设备',
    '车辆', '规划', '设计', '咨询', '监理', '管理'
]
```

**合规能力关键词**:
```python
compliance_keywords = [
    '教育', '教学', '学校', '培训', '医疗', '医院', '卫生',
    '健康', '疾控', '安防', '监控', '公安', '执法', '环保',
    '养老', '福利', '社区'
]
```

#### 4.2.2 行业映射规则

| 行业类别 | 映射能力类型 |
|----------|--------------|
| 金融业、财政、税务 | 汲取能力 |
| 建筑业、交通运输、信息传输 | 协调能力 |
| 教育、医疗卫生、社会工作 | 合规能力 |

#### 4.2.3 标注置信度计算

```python
confidence = max_score / (total_score + 1)
```

- `max_score`: 最高类别的关键词匹配数
- `total_score`: 所有类别关键词匹配总数

### 4.3 特征工程

#### 4.3.1 文本预处理
1. 合并多个文本字段（合同名称、主要标的名称、所属行业）
2. 去除特殊字符，保留中英文和数字
3. 去除多余空格

#### 4.3.2 特征提取方法

**TF-IDF向量化**:
- `max_features`: 3000
- `ngram_range`: (1, 2) - 单字和双字组合
- `min_df`: 2 - 最小文档频率
- `max_df`: 0.95 - 最大文档频率

### 4.4 机器学习模型

本项目训练并比较了以下6种模型：

| 模型 | 说明 | 关键参数 |
|------|------|----------|
| Logistic Regression | 逻辑回归 | `class_weight='balanced'` |
| Random Forest | 随机森林 | `n_estimators=100, max_depth=20` |
| SVM | 支持向量机 | `kernel='linear'` |
| Naive Bayes | 朴素贝叶斯 | `alpha=0.1` |
| Gradient Boosting | 梯度提升 | `n_estimators=100, max_depth=5` |
| MLP Neural Network | 多层感知机 | `hidden_layer_sizes=(256, 128)` |

### 4.5 模型评估指标

- **准确率 (Accuracy)**: 正确分类的比例
- **F1-Macro**: 各类别F1分数的宏平均
- **混淆矩阵**: 各类别的预测情况
- **5折交叉验证**: 评估模型稳定性

---

## 5. 技术实现

### 5.1 环境依赖

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 5.2 项目文件结构

```
MOR-SI/
├── 2012.dta                    # 2012年数据
├── 2013.dta                    # 2013年数据
├── 2014.dta                    # 2014年数据
├── annurev-polisci-*.pdf       # 参考论文
├── state_capacity_classifier.py # 分类器主程序
├── README_StateCapacity_Classification.md  # 本文档
│
# 运行后生成的文件:
├── classified_contracts.csv    # 分类结果
├── model_evaluation.csv        # 模型评估结果
├── label_distribution.png      # 标签分布图
├── model_comparison.png        # 模型对比图
└── confusion_matrix.png        # 混淆矩阵图
```

### 5.3 核心类说明

#### StateCapacityLabeler
```python
class StateCapacityLabeler:
    """基于规则的标注器"""

    def label_single(self, contract_name, subject_name, industry):
        """对单条记录进行标注，返回(标签, 置信度, 原因)"""

    def label_dataframe(self, df):
        """对整个数据框进行标注"""
```

#### StateCapacityClassifier
```python
class StateCapacityClassifier:
    """机器学习分类器"""

    def prepare_data(self, df, test_size=0.2):
        """准备训练和测试数据"""

    def train_models(self, X_train, y_train):
        """训练多个模型"""

    def evaluate(self, X_test, y_test):
        """评估所有模型"""

    def predict(self, texts):
        """预测新文本"""
```

---

## 6. 使用指南

### 6.1 运行分类器

```bash
cd /home/user/MOR-SI
python3 state_capacity_classifier.py
```

### 6.2 预期输出

程序将依次执行：
1. 加载数据
2. 基于规则的初始标注
3. 生成可视化图表
4. 训练机器学习模型
5. 评估模型性能
6. 保存分类结果

### 6.3 使用分类器进行预测

```python
from state_capacity_classifier import *

# 加载数据并训练
df = load_data()
labeler = StateCapacityLabeler()
df_labeled = labeler.label_dataframe(df)

classifier = StateCapacityClassifier()
X_train, X_test, y_train, y_test = classifier.prepare_data(df_labeled)
classifier.train_models(X_train, y_train)

# 预测新合同
new_contracts = [
    "税务信息系统升级改造",
    "小学多媒体教学设备采购",
    "市政道路维修工程"
]
labels, probs = classifier.predict(new_contracts)
print(labels)  # ['extractive', 'compliance', 'coordination']
```

---

## 7. 结果解读

### 7.1 标签分布预期

根据数据特征，预期分布：
- **合规能力 (Compliance)**: ~45% - 教育医疗类占主导
- **协调能力 (Coordination)**: ~40% - 通用政府采购和基建
- **汲取能力 (Extractive)**: ~15% - 财税相关较少

### 7.2 模型性能预期

| 指标 | 预期范围 |
|------|----------|
| 准确率 | 0.65 - 0.80 |
| F1-Macro | 0.60 - 0.75 |

### 7.3 分类示例解读

| 合同名称 | 分类结果 | 解读 |
|----------|----------|------|
| 税务系统升级改造项目 | 汲取能力 | 与国家获取资源的财税系统相关 |
| 小学教学设备采购 | 合规能力 | 教育服务提供，确保教育政策执行 |
| 市政道路维修工程 | 协调能力 | 基础设施建设，组织公共服务 |
| 医院CT设备采购 | 合规能力 | 医疗服务提供 |
| 办公家具采购 | 协调能力 | 行政办公协调 |

---

## 8. 局限性与展望

### 8.1 当前局限性

1. **伪标签噪声**: 规则标注可能引入系统性偏差
2. **类别不平衡**: 汲取能力类别样本较少
3. **语义理解有限**: TF-IDF无法捕捉深层语义
4. **数据范围**: 仅限于2012-2014年，地域集中在重庆

### 8.2 改进方向

1. **引入人工标注**: 对部分样本进行人工审核
2. **深度学习模型**: 使用BERT等预训练模型
3. **多任务学习**: 结合行业分类和能力分类
4. **扩展数据**: 纳入更多年份和地域的数据

### 8.3 潜在应用

1. **政策研究**: 分析政府采购与国家能力建设的关系
2. **区域比较**: 不同地区的能力结构差异
3. **时序分析**: 国家能力的演变趋势
4. **效率评估**: 采购金额与能力建设的投入产出分析

---

## 参考文献

1. Berwick, E., & Christia, F. (2018). State Capacity Redux: Integrating Classical and Experimental Contributions to an Enduring Debate. *Annual Review of Political Science*, 21, 71-91.

2. Tilly, C. (1990). Coercion, Capital, and European States, AD 990-1990.

3. Levi, M. (1988). Of Rule and Revenue.

4. Evans, P. (1989). Predatory, developmental and other apparatuses.

5. Bjorkman, M., & Svensson, J. (2009). Power to the people: Evidence from a randomized field experiment on community-based monitoring in Uganda.

---

## 附录

### A. 完整关键词列表

详见 `state_capacity_classifier.py` 中的 `StateCapacityLabeler` 类定义。

### B. 模型超参数

详见 `state_capacity_classifier.py` 中的 `train_models` 方法。

---

**文档版本**: 1.0
**最后更新**: 2025-11-28
**作者**: Claude AI
