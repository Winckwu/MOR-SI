#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - LLM标注 + ML模型方案
基于 Berwick & Christia (2018) "State Capacity Redux" 论文框架

工作流程:
1. LLM语义规则标注全量数据
2. 筛选高置信度样本作为训练集
3. 训练SVM模型
4. 保存模型供新数据集使用
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：LLM语义标注器
# 模拟LLM的语义理解能力，基于多维度特征进行分类
# ============================================================

class LLMSemanticLabeler:
    """
    LLM语义标注器

    设计思路：模拟LLM的多层次语义理解能力
    1. 机构识别层：识别采购单位的性质
    2. 核心概念层：匹配理论框架的核心概念
    3. 活动类型层：识别具体业务活动
    4. 上下文推理层：综合多个信号进行推理
    """

    def __init__(self):
        self._init_semantic_rules()
        self._init_context_rules()

    def _init_semantic_rules(self):
        """初始化语义规则库"""
        self.semantic_rules = {
            "汲取能力": {
                "description": "国家获取资源的能力 (Extractive Capacity)",
                "theoretical_basis": [
                    "Tilly (1990): 税收是国家形成的核心",
                    "Levi (1988): 国家作为收入最大化者",
                    "D'Arcy & Nistotskaya (2016): 地籍调查与财产税",
                    "Brambor et al. (2016): 统计能力测量"
                ],
                # 核心概念词 - 直接反映汲取能力
                "core_concepts": [
                    "税", "税收", "税务", "征税", "纳税",
                    "财政", "预算", "收入", "征收",
                    "审计", "稽查", "核算"
                ],
                # 机构特征词 - 执行汲取功能的机构
                "org_keywords": [
                    "税务局", "国税", "地税", "税务所",
                    "财政局", "财政厅", "财政所",
                    "海关", "关税",
                    "统计局", "统计站",
                    "审计局", "审计署",
                    "国土局", "土地局", "不动产"
                ],
                # 活动特征词 - 与汲取相关的活动
                "activity_keywords": [
                    "土地调查", "地籍", "确权", "登记",
                    "普查", "统计", "调查",
                    "评估", "估价", "核定",
                    "征管", "稽核", "票据"
                ],
                # 权重系数
                "weight": 1.5
            },

            "协调能力": {
                "description": "国家组织集体行动的能力 (Coordination Capacity)",
                "theoretical_basis": [
                    "Mann (1984, 2008): 基础设施权力",
                    "Weber (1946): 官僚制与行政能力",
                    "Evans (1989): 嵌入式自主",
                    "Herbst (2000): 地理渗透与领土控制"
                ],
                # 核心概念词
                "core_concepts": [
                    "建设", "工程", "施工", "修建",
                    "规划", "改造", "扩建", "新建",
                    "网络", "信息化", "数字化", "系统",
                    "基础设施"
                ],
                # 机构特征词
                "org_keywords": [
                    "建设局", "住建局", "规划局", "城建",
                    "交通局", "公路局", "铁路",
                    "水利局", "水务局",
                    "电力", "供电", "电网",
                    "通信", "电信", "移动", "联通",
                    "发改委", "发展改革"
                ],
                # 活动特征词
                "activity_keywords": [
                    "公路", "道路", "桥梁", "隧道",
                    "供水", "排水", "供电", "供气", "供热",
                    "办公", "机关", "政府采购",
                    "信息系统", "平台建设", "网站"
                ],
                "weight": 1.0
            },

            "合规能力": {
                "description": "确保公民和官僚服从国家目标的能力 (Compliance Capacity)",
                "theoretical_basis": [
                    "Olken (2007): 监督与腐败实验",
                    "Muralidharan & Sundararaman (2011): 教师激励",
                    "Bjorkman & Svensson (2009): 社区监督医疗",
                    "Holland (2015): 执法选择性"
                ],
                # 核心概念词
                "core_concepts": [
                    "教育", "教学", "培训", "学习",
                    "医疗", "卫生", "健康", "防疫",
                    "监督", "检查", "执法", "监察",
                    "服务", "保障"
                ],
                # 机构特征词
                "org_keywords": [
                    "学校", "大学", "学院", "中学", "小学", "幼儿园",
                    "医院", "卫生院", "诊所", "疾控",
                    "教育局", "教育厅", "教体局",
                    "卫生局", "卫健委", "卫生厅",
                    "公安局", "派出所", "消防", "武警",
                    "法院", "检察院", "司法局",
                    "市场监管", "质监", "药监", "食药监"
                ],
                # 活动特征词
                "activity_keywords": [
                    "教学设备", "实验室", "图书", "课桌",
                    "医疗设备", "药品", "器械", "诊断",
                    "执法装备", "警用", "安防", "监控",
                    "培训项目", "考核", "评估",
                    "社保", "民政", "救助", "福利"
                ],
                "weight": 1.2
            }
        }

    def _init_context_rules(self):
        """初始化上下文推理规则"""
        # 机构名称优先级规则
        self.org_priority = {
            # 汲取能力机构
            "税务": ("汲取能力", 5),
            "国税": ("汲取能力", 5),
            "地税": ("汲取能力", 5),
            "财政": ("汲取能力", 4),
            "海关": ("汲取能力", 5),
            "审计": ("汲取能力", 4),
            "统计局": ("汲取能力", 4),
            "国土": ("汲取能力", 3),

            # 合规能力机构
            "学校": ("合规能力", 4),
            "大学": ("合规能力", 4),
            "学院": ("合规能力", 4),
            "中学": ("合规能力", 4),
            "小学": ("合规能力", 4),
            "幼儿园": ("合规能力", 4),
            "医院": ("合规能力", 4),
            "卫生": ("合规能力", 3),
            "教育局": ("合规能力", 4),
            "教育厅": ("合规能力", 4),
            "公安": ("合规能力", 4),
            "消防": ("合规能力", 4),
            "法院": ("合规能力", 4),
            "检察": ("合规能力", 4),

            # 协调能力机构
            "交通": ("协调能力", 3),
            "公路": ("协调能力", 3),
            "水利": ("协调能力", 3),
            "电力": ("协调能力", 3),
            "建设局": ("协调能力", 3),
            "住建": ("协调能力", 3),
        }

        # 项目类型修正规则
        self.project_modifiers = {
            "建设工程": {"协调能力": 2},
            "改造工程": {"协调能力": 2},
            "施工项目": {"协调能力": 2},
            "采购项目": {},  # 中性，取决于具体内容
            "服务项目": {"合规能力": 1},
            "培训项目": {"合规能力": 2},
        }

    def label(self, text):
        """
        对单条合同进行LLM语义标注

        返回: (标签, 置信度, 匹配详情)
        """
        if pd.isna(text) or not text:
            return "协调能力", 0.33, {"reason": "空文本默认分类"}

        text = str(text)
        scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}
        details = {"org_matches": [], "core_matches": [], "activity_matches": []}

        # 第1层：机构识别（最高优先级）
        for org_keyword, (capacity, weight) in self.org_priority.items():
            if org_keyword in text:
                scores[capacity] += weight
                details["org_matches"].append(f"{org_keyword}→{capacity}(+{weight})")

        # 第2层：核心概念匹配
        for capacity, rules in self.semantic_rules.items():
            cap_weight = rules["weight"]

            for word in rules["core_concepts"]:
                if word in text:
                    scores[capacity] += 2 * cap_weight
                    details["core_matches"].append(f"{word}→{capacity}")

            # 第3层：机构特征匹配
            for word in rules["org_keywords"]:
                if word in text:
                    scores[capacity] += 1.5 * cap_weight

            # 第4层：活动特征匹配
            for word in rules["activity_keywords"]:
                if word in text:
                    scores[capacity] += 1 * cap_weight
                    details["activity_matches"].append(f"{word}→{capacity}")

        # 第5层：项目类型修正
        for proj_type, modifiers in self.project_modifiers.items():
            if proj_type in text:
                for cap, mod_weight in modifiers.items():
                    scores[cap] += mod_weight

        # 计算最终结果
        total = sum(scores.values())
        if total == 0:
            return "协调能力", 0.33, {"reason": "无匹配关键词，默认分类"}

        max_capacity = max(scores, key=scores.get)
        confidence = scores[max_capacity] / total

        # 如果最高分和次高分接近，降低置信度
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            if margin < 0.3:
                confidence *= 0.8  # 边界模糊时降低置信度

        details["scores"] = scores
        details["reason"] = f"综合得分: {scores}"

        return max_capacity, confidence, details

    def label_batch(self, texts, show_progress=True):
        """批量标注"""
        results = []
        total = len(texts)

        for idx, text in enumerate(texts):
            label, conf, details = self.label(text)
            results.append({
                "text": text,
                "label": label,
                "confidence": conf,
                "details": details
            })

            if show_progress and (idx + 1) % 5000 == 0:
                print(f"  已标注: {idx + 1}/{total}")

        return results


# ============================================================
# 第二部分：分词器（模块级别定义，支持pickle）
# ============================================================

def simple_tokenizer(text):
    """
    简单的中文分词器（字符级 + bigram）
    必须在模块级别定义以支持pickle序列化
    """
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


# ============================================================
# 第三部分：ML模型训练器
# ============================================================

class LLMMLClassifier:
    """
    LLM标注 + ML模型分类器
    """

    def __init__(self, confidence_threshold=0.6):
        self.labeler = LLMSemanticLabeler()
        self.confidence_threshold = confidence_threshold
        self.vectorizer = None
        self.model = None
        self.label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
        self.label_map_reverse = {0: "汲取能力", 1: "协调能力", 2: "合规能力"}

    def label_data(self, df, text_column="合同名称"):
        """使用LLM标注数据"""
        print("=" * 70)
        print("第1步：LLM语义标注")
        print("=" * 70)

        texts = df[text_column].tolist()
        print(f"总样本数: {len(texts)}")

        results = self.labeler.label_batch(texts)

        # 统计标注结果
        all_labels = [r["label"] for r in results]
        all_confs = [r["confidence"] for r in results]

        print(f"\n标注完成！")
        print(f"\n标注分布:")
        for label in ["汲取能力", "协调能力", "合规能力"]:
            count = all_labels.count(label)
            print(f"  {label}: {count} ({count/len(all_labels)*100:.1f}%)")

        print(f"\n置信度分布:")
        print(f"  平均置信度: {np.mean(all_confs):.3f}")
        print(f"  中位数置信度: {np.median(all_confs):.3f}")
        print(f"  高置信度(≥{self.confidence_threshold}): {sum(1 for c in all_confs if c >= self.confidence_threshold)} ({sum(1 for c in all_confs if c >= self.confidence_threshold)/len(all_confs)*100:.1f}%)")

        return results

    def prepare_training_data(self, labeling_results):
        """准备训练数据"""
        print("\n" + "=" * 70)
        print("第2步：准备训练数据")
        print("=" * 70)

        # 筛选高置信度样本
        high_conf_samples = [
            r for r in labeling_results
            if r["confidence"] >= self.confidence_threshold and r["text"]
        ]

        print(f"高置信度样本(≥{self.confidence_threshold}): {len(high_conf_samples)}")

        # 按类别统计
        print(f"\n训练集类别分布:")
        for label in ["汲取能力", "协调能力", "合规能力"]:
            count = sum(1 for r in high_conf_samples if r["label"] == label)
            print(f"  {label}: {count} ({count/len(high_conf_samples)*100:.1f}%)")

        texts = [r["text"] for r in high_conf_samples]
        labels = [self.label_map[r["label"]] for r in high_conf_samples]

        return texts, labels

    def train(self, texts, labels, test_size=0.2):
        """训练ML模型"""
        print("\n" + "=" * 70)
        print("第3步：训练ML模型")
        print("=" * 70)

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"训练集: {len(X_train)} 条")
        print(f"测试集: {len(X_test)} 条")

        # TF-IDF向量化
        print("\n正在进行TF-IDF向量化...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=simple_tokenizer,
            max_features=5000,
            ngram_range=(1, 2)
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"特征维度: {X_train_tfidf.shape[1]}")

        # 训练多个模型进行对比
        models = {
            "SVM": LinearSVC(max_iter=10000, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        print("\n模型训练与评估:")
        print("-" * 50)

        best_model = None
        best_score = 0
        best_name = ""
        results = {}

        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results[name] = {"accuracy": acc, "f1": f1, "predictions": y_pred}
            print(f"  {name}: 准确率={acc:.4f}, F1={f1:.4f}")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        print(f"\n最佳模型: {best_name} (F1={best_score:.4f})")
        self.model = best_model

        # 详细评估最佳模型
        print("\n" + "=" * 70)
        print(f"最佳模型详细评估 ({best_name})")
        print("=" * 70)

        y_pred = results[best_name]["predictions"]

        print("\n分类报告:")
        print(classification_report(
            y_test, y_pred,
            target_names=["汲取能力", "协调能力", "合规能力"]
        ))

        print("混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"           预测→  汲取    协调    合规")
        print(f"  真实↓")
        for i, label in enumerate(["汲取能力", "协调能力", "合规能力"]):
            print(f"  {label}    {cm[i][0]:5d}   {cm[i][1]:5d}   {cm[i][2]:5d}")

        return {
            "best_model": best_name,
            "accuracy": results[best_name]["accuracy"],
            "f1": results[best_name]["f1"],
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
            "all_results": results
        }

    def predict(self, texts):
        """预测新数据"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train方法")

        X_tfidf = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_tfidf)

        results = []
        for text, pred in zip(texts, predictions):
            results.append({
                "text": text,
                "predicted_label": self.label_map_reverse[pred]
            })

        return results

    def save_model(self, filepath):
        """保存模型"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练")

        saved_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_map": self.label_map,
            "label_map_reverse": self.label_map_reverse,
            "confidence_threshold": self.confidence_threshold
        }

        with open(filepath, 'wb') as f:
            pickle.dump(saved_data, f)

        print(f"模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)

        classifier = cls()
        classifier.model = saved_data["model"]
        classifier.vectorizer = saved_data["vectorizer"]
        classifier.label_map = saved_data["label_map"]
        classifier.label_map_reverse = saved_data["label_map_reverse"]
        classifier.confidence_threshold = saved_data.get("confidence_threshold", 0.6)

        print(f"模型已加载: {filepath}")
        return classifier


# ============================================================
# 第四部分：完整流程执行
# ============================================================

def run_full_pipeline(data_path, text_column="合同名称", output_dir="."):
    """
    运行完整的LLM标注→ML训练流程
    """
    print("=" * 70)
    print("国家能力合同分类 - LLM标注 + ML模型方案")
    print("基于 Berwick & Christia (2018) 理论框架")
    print("=" * 70)
    print(f"\n数据文件: {data_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("\n加载数据...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_excel(data_path)

    print(f"数据集大小: {len(df)} 条")

    # 创建分类器
    classifier = LLMMLClassifier(confidence_threshold=0.6)

    # 步骤1：LLM标注
    labeling_results = classifier.label_data(df, text_column)

    # 步骤2：准备训练数据
    texts, labels = classifier.prepare_training_data(labeling_results)

    # 步骤3：训练模型
    eval_results = classifier.train(texts, labels)

    # 步骤4：对全量数据进行预测
    print("\n" + "=" * 70)
    print("第4步：全量数据预测")
    print("=" * 70)

    all_texts = [str(t) if t else "" for t in df[text_column].tolist()]
    all_predictions = classifier.predict(all_texts)

    # 添加预测结果到DataFrame
    df["LLM标注"] = [r["label"] for r in labeling_results]
    df["LLM置信度"] = [r["confidence"] for r in labeling_results]
    df["ML预测"] = [r["predicted_label"] for r in all_predictions]

    # 统计预测结果
    print("\n全量数据预测分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = sum(1 for r in all_predictions if r["predicted_label"] == label)
        print(f"  {label}: {count} ({count/len(all_predictions)*100:.1f}%)")

    # 保存结果
    print("\n" + "=" * 70)
    print("第5步：保存结果")
    print("=" * 70)

    # 保存分类结果
    result_path = os.path.join(output_dir, "LLM_ML分类结果.xlsx")
    df.to_excel(result_path, index=False)
    print(f"分类结果已保存: {result_path}")

    # 保存模型
    model_path = os.path.join(output_dir, "llm_ml_model.pkl")
    classifier.save_model(model_path)

    # 生成分析报告
    report = generate_analysis_report(
        df, labeling_results, eval_results, classifier, output_dir
    )

    return classifier, df, eval_results


def generate_analysis_report(df, labeling_results, eval_results, classifier, output_dir):
    """生成详细分析报告"""
    print("\n" + "=" * 70)
    print("第6步：生成分析报告")
    print("=" * 70)

    report_path = os.path.join(output_dir, "LLM_ML分类分析报告.md")

    # 计算各项统计
    all_labels = [r["label"] for r in labeling_results]
    all_confs = [r["confidence"] for r in labeling_results]
    high_conf_count = sum(1 for c in all_confs if c >= classifier.confidence_threshold)

    # 各类别置信度
    conf_by_label = {}
    for label in ["汲取能力", "协调能力", "合规能力"]:
        confs = [r["confidence"] for r in labeling_results if r["label"] == label]
        if confs:
            conf_by_label[label] = {
                "mean": np.mean(confs),
                "median": np.median(confs),
                "std": np.std(confs),
                "high_ratio": sum(1 for c in confs if c >= 0.8) / len(confs)
            }

    report = f"""# 国家能力合同分类 - LLM标注+ML模型 分析报告

**基于 Berwick & Christia (2018) "State Capacity Redux" 论文框架**

---

## 第一部分：方法论概述

### 1.1 研究框架

本分类器基于Berwick & Christia (2018)提出的国家能力三维框架：

| 能力类型 | 英文名称 | 核心定义 | 理论来源 |
|----------|----------|----------|----------|
| 汲取能力 | Extractive Capacity | 国家从社会获取资源的能力 | Tilly (1990), Levi (1988) |
| 协调能力 | Coordination Capacity | 国家组织集体行动、提供公共品的能力 | Mann (1984), Weber (1946) |
| 合规能力 | Compliance Capacity | 确保公民和官僚服从国家目标的能力 | Olken (2007), Holland (2015) |

### 1.2 技术方案

采用**LLM语义标注 + ML模型**的两阶段方案：

```
阶段1: LLM语义标注
  ├─ 多层次语义规则
  │   ├─ 机构识别层（识别采购单位性质）
  │   ├─ 核心概念层（匹配理论框架核心概念）
  │   ├─ 活动类型层（识别具体业务活动）
  │   └─ 上下文推理层（综合多信号推理）
  └─ 输出: 标签 + 置信度

阶段2: ML模型训练
  ├─ 筛选高置信度样本（≥60%）
  ├─ TF-IDF特征提取
  ├─ SVM/LR/RF模型训练
  └─ 输出: 可复用的分类模型
```

---

## 第二部分：数据概况

### 2.1 原始数据

| 项目 | 数值 |
|------|------|
| 数据来源 | 2015年政府采购合同 |
| 总记录数 | {len(df):,} 条 |
| 文本字段 | 合同名称 |

### 2.2 合同名称统计

| 指标 | 数值 |
|------|------|
| 有效记录 | {len(df):,} 条 |
| 平均长度 | {df['合同名称'].astype(str).str.len().mean():.1f} 字符 |
| 最短长度 | {df['合同名称'].astype(str).str.len().min()} 字符 |
| 最长长度 | {df['合同名称'].astype(str).str.len().max()} 字符 |

---

## 第三部分：LLM语义标注结果

### 3.1 标注分布

| 能力类型 | 数量 | 占比 |
|----------|------|------|
| 汲取能力 | {all_labels.count("汲取能力"):,} | {all_labels.count("汲取能力")/len(all_labels)*100:.1f}% |
| 协调能力 | {all_labels.count("协调能力"):,} | {all_labels.count("协调能力")/len(all_labels)*100:.1f}% |
| 合规能力 | {all_labels.count("合规能力"):,} | {all_labels.count("合规能力")/len(all_labels)*100:.1f}% |

### 3.2 置信度分析

#### 总体置信度

| 指标 | 数值 |
|------|------|
| 平均置信度 | {np.mean(all_confs):.3f} |
| 中位数置信度 | {np.median(all_confs):.3f} |
| 标准差 | {np.std(all_confs):.3f} |
| 高置信度样本(≥0.6) | {high_conf_count:,} ({high_conf_count/len(all_confs)*100:.1f}%) |

#### 各类别置信度

| 能力类型 | 平均置信度 | 中位数 | 高置信度(≥0.8)占比 |
|----------|------------|--------|-------------------|
| 汲取能力 | {conf_by_label.get("汲取能力", {}).get("mean", 0):.3f} | {conf_by_label.get("汲取能力", {}).get("median", 0):.3f} | {conf_by_label.get("汲取能力", {}).get("high_ratio", 0)*100:.1f}% |
| 协调能力 | {conf_by_label.get("协调能力", {}).get("mean", 0):.3f} | {conf_by_label.get("协调能力", {}).get("median", 0):.3f} | {conf_by_label.get("协调能力", {}).get("high_ratio", 0)*100:.1f}% |
| 合规能力 | {conf_by_label.get("合规能力", {}).get("mean", 0):.3f} | {conf_by_label.get("合规能力", {}).get("median", 0):.3f} | {conf_by_label.get("合规能力", {}).get("high_ratio", 0)*100:.1f}% |

---

## 第四部分：ML模型训练结果

### 4.1 训练数据

| 项目 | 数值 |
|------|------|
| 高置信度训练样本 | {high_conf_count:,} 条 |
| 训练集占比 | 80% |
| 测试集占比 | 20% |
| 置信度阈值 | ≥0.6 |

### 4.2 模型对比

| 模型 | 准确率 | F1分数 |
|------|--------|--------|
| SVM | {eval_results["all_results"]["SVM"]["accuracy"]:.4f} | {eval_results["all_results"]["SVM"]["f1"]:.4f} |
| Logistic Regression | {eval_results["all_results"]["Logistic Regression"]["accuracy"]:.4f} | {eval_results["all_results"]["Logistic Regression"]["f1"]:.4f} |
| Random Forest | {eval_results["all_results"]["Random Forest"]["accuracy"]:.4f} | {eval_results["all_results"]["Random Forest"]["f1"]:.4f} |

**最佳模型: {eval_results["best_model"]}**

### 4.3 最佳模型详细评估

#### 混淆矩阵

| 真实\\预测 | 汲取能力 | 协调能力 | 合规能力 |
|-----------|----------|----------|----------|
| 汲取能力 | {eval_results["confusion_matrix"][0][0]} | {eval_results["confusion_matrix"][0][1]} | {eval_results["confusion_matrix"][0][2]} |
| 协调能力 | {eval_results["confusion_matrix"][1][0]} | {eval_results["confusion_matrix"][1][1]} | {eval_results["confusion_matrix"][1][2]} |
| 合规能力 | {eval_results["confusion_matrix"][2][0]} | {eval_results["confusion_matrix"][2][1]} | {eval_results["confusion_matrix"][2][2]} |

#### 性能指标

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | {eval_results["accuracy"]:.4f} |
| 宏平均F1 (Macro F1) | {eval_results["f1"]:.4f} |

---

## 第五部分：全量预测结果

### 5.1 预测分布

| 能力类型 | 数量 | 占比 |
|----------|------|------|
| 汲取能力 | {(df["ML预测"]=="汲取能力").sum():,} | {(df["ML预测"]=="汲取能力").sum()/len(df)*100:.1f}% |
| 协调能力 | {(df["ML预测"]=="协调能力").sum():,} | {(df["ML预测"]=="协调能力").sum()/len(df)*100:.1f}% |
| 合规能力 | {(df["ML预测"]=="合规能力").sum():,} | {(df["ML预测"]=="合规能力").sum()/len(df)*100:.1f}% |

### 5.2 典型样本

#### 汲取能力样本
"""

    # 添加典型样本
    for label in ["汲取能力", "协调能力", "合规能力"]:
        samples = df[df["ML预测"] == label].head(3)
        for _, row in samples.iterrows():
            report += f"\n- {row['合同名称'][:60]}..."
        report += f"\n\n#### {['协调能力', '合规能力'][['汲取能力', '协调能力', '合规能力'].index(label) - 2] if label != '合规能力' else ''}"

    report += f"""

---

## 第六部分：LLM语义规则详解

### 6.1 汲取能力关键词

| 类别 | 关键词 | 理论来源 |
|------|--------|----------|
| 核心概念 | 税、税收、财政、预算、审计 | Tilly (1990), Levi (1988) |
| 机构特征 | 税务局、财政局、海关、统计局 | D'Arcy & Nistotskaya (2016) |
| 活动特征 | 土地调查、确权、登记、普查 | Brambor et al. (2016) |

### 6.2 协调能力关键词

| 类别 | 关键词 | 理论来源 |
|------|--------|----------|
| 核心概念 | 建设、工程、规划、网络、信息化 | Mann (1984, 2008) |
| 机构特征 | 建设局、交通局、水利局、电力 | Weber (1946), Evans (1989) |
| 活动特征 | 公路、桥梁、供水、供电、办公 | Herbst (2000) |

### 6.3 合规能力关键词

| 类别 | 关键词 | 理论来源 |
|------|--------|----------|
| 核心概念 | 教育、医疗、监督、执法、培训 | Olken (2007) |
| 机构特征 | 学校、医院、公安局、法院 | Muralidharan (2011) |
| 活动特征 | 教学设备、医疗设备、执法装备 | Holland (2015) |

---

## 第七部分：模型使用说明

### 7.1 输出文件

| 文件 | 说明 |
|------|------|
| `llm_ml_model.pkl` | 训练好的ML模型（可直接加载使用） |
| `LLM_ML分类结果.xlsx` | 全量数据分类结果 |
| `LLM_ML分类分析报告.md` | 本分析报告 |
| `llm_ml_classifier.py` | 分类器源代码 |

### 7.2 新数据预测代码

```python
from llm_ml_classifier import LLMMLClassifier, simple_tokenizer
import pandas as pd

# 加载模型
classifier = LLMMLClassifier.load_model("llm_ml_model.pkl")

# 读取新数据
new_df = pd.read_excel("新数据.xlsx")

# 预测
texts = new_df["合同名称"].astype(str).tolist()
results = classifier.predict(texts)

# 添加预测结果
new_df["预测类别"] = [r["predicted_label"] for r in results]

# 保存结果
new_df.to_excel("预测结果.xlsx", index=False)
```

### 7.3 命令行使用

```bash
# 训练新模型
python llm_ml_classifier.py --train --data 数据.xlsx

# 预测新数据
python llm_ml_classifier.py --predict --data 新数据.xlsx --model llm_ml_model.pkl
```

---

## 第八部分：方法局限与改进方向

### 8.1 当前局限

1. **语义理解深度有限**: 模拟LLM的规则无法达到真实LLM的语义理解能力
2. **多标签问题**: 部分合同可能同时涉及多种能力，当前采用单标签分类
3. **上下文信息未充分利用**: 未使用采购单位、金额等元数据

### 8.2 改进方向

1. 接入真实LLM API（OpenAI/Claude）进行标注
2. 引入BERT等预训练语言模型
3. 采用多标签分类框架
4. 结合元数据构建多模态分类器

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"分析报告已保存: {report_path}")

    return report


# ============================================================
# 第五部分：命令行接口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="国家能力合同分类器 - LLM标注+ML模型")
    parser.add_argument("--train", action="store_true", help="训练新模型")
    parser.add_argument("--predict", action="store_true", help="预测新数据")
    parser.add_argument("--data", type=str, help="数据文件路径")
    parser.add_argument("--model", type=str, default="llm_ml_model.pkl", help="模型文件路径")
    parser.add_argument("--column", type=str, default="合同名称", help="文本列名")
    parser.add_argument("--output", type=str, default=".", help="输出目录")

    args = parser.parse_args()

    if args.train:
        if not args.data:
            print("错误: 请指定数据文件 --data")
        else:
            run_full_pipeline(args.data, args.column, args.output)

    elif args.predict:
        if not args.data:
            print("错误: 请指定数据文件 --data")
        else:
            # 加载模型
            classifier = LLMMLClassifier.load_model(args.model)

            # 读取数据
            df = pd.read_excel(args.data)
            texts = df[args.column].astype(str).tolist()

            # 预测
            results = classifier.predict(texts)

            # 保存结果
            df["预测类别"] = [r["predicted_label"] for r in results]
            output_path = args.data.replace(".xlsx", "_预测结果.xlsx").replace(".xls", "_预测结果.xlsx")
            df.to_excel(output_path, index=False)

            print(f"预测完成，结果已保存: {output_path}")

    else:
        # 默认运行完整流程
        run_full_pipeline("/home/user/MOR-SI/2015.xls", "合同名称", "/home/user/MOR-SI")
