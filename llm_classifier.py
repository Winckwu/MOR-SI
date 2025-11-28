#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - LLM版本
使用大语言模型进行分类

两种方案对比:
1. LLM标注训练集 → 训练ML模型 → 预测 (成本低，可扩展)
2. LLM直接分类所有数据 (可能更准，但成本高)
"""

import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LLM分类提示词
# ============================================================

CLASSIFICATION_PROMPT = """你是一个政治学专家，需要根据Berwick & Christia (2018)的国家能力理论框架，将政府采购合同分类为三种国家能力之一。

## 三种国家能力定义：

1. **汲取能力 (Extractive Capacity)**: 国家获取资源的能力
   - 关键特征: 税收、财政、预算、土地调查、统计普查、资产评估、海关
   - 例子: 税务局设备采购、土地确权项目、财政预算系统

2. **协调能力 (Coordination Capacity)**: 国家组织集体行动的能力
   - 关键特征: 基础设施建设、公路铁路、水利电力、网络通信、行政办公、城市规划
   - 例子: 公路建设工程、电网改造项目、政府办公设备

3. **合规能力 (Compliance Capacity)**: 确保公民和官僚服从国家目标的能力
   - 关键特征: 监督检查、教育培训、医疗卫生、社会保障、执法安防
   - 例子: 学校教学设备、医院医疗器械、消防安防设备

## 请对以下合同进行分类:

合同名称: {contract_name}

请只回答一个类别名称: 汲取能力、协调能力、合规能力
"""

# ============================================================
# 模拟LLM分类（基于增强规则，模拟LLM的语义理解）
# ============================================================

class SimulatedLLMClassifier:
    """
    模拟LLM分类器
    使用更复杂的语义规则来模拟LLM的理解能力
    """

    def __init__(self):
        # 更细粒度的语义规则
        self.semantic_rules = {
            "汲取能力": {
                # 核心概念
                "core": ["税", "财政", "预算", "收入", "征收", "审计"],
                # 机构特征
                "org": ["税务局", "财政局", "国税", "地税", "海关", "统计局"],
                # 活动特征
                "activity": ["土地调查", "确权", "登记", "普查", "统计", "评估", "审计"],
                # 权重
                "weight": 1.5
            },
            "协调能力": {
                "core": ["建设", "工程", "规划", "改造", "网络", "基础设施"],
                "org": ["建设局", "交通局", "水利局", "电力", "通信", "住建"],
                "activity": ["公路", "铁路", "桥梁", "供水", "供电", "信息化", "数字化"],
                "weight": 1.0
            },
            "合规能力": {
                "core": ["教育", "医疗", "培训", "监督", "服务", "执法"],
                "org": ["学校", "医院", "大学", "中学", "小学", "幼儿园", "卫生", "公安", "消防"],
                "activity": ["教学", "医疗", "培训", "考核", "监察", "巡查", "防疫"],
                "weight": 1.2
            }
        }

        # 上下文增强规则
        self.context_rules = {
            # 机构名称优先级
            "org_priority": {
                "税务": "汲取能力",
                "财政": "汲取能力",
                "海关": "汲取能力",
                "国土": "汲取能力",
                "教育局": "合规能力",
                "卫生": "合规能力",
                "医院": "合规能力",
                "学校": "合规能力",
                "公安": "合规能力",
                "交通": "协调能力",
                "水利": "协调能力",
                "电力": "协调能力",
            },
            # 项目类型
            "project_type": {
                "建设工程": "协调能力",
                "改造工程": "协调能力",
                "采购项目": None,  # 需要进一步判断
                "服务项目": None,
            }
        }

    def classify(self, text):
        """模拟LLM分类"""
        if pd.isna(text) or not text:
            return "协调能力", 0.33, "default"

        text = str(text)
        scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}
        reasons = []

        # 1. 检查机构名称优先级
        for org, capacity in self.context_rules["org_priority"].items():
            if org in text:
                scores[capacity] += 3
                reasons.append(f"机构'{org}'→{capacity}")

        # 2. 语义规则匹配
        for capacity, rules in self.semantic_rules.items():
            # 核心词匹配
            for word in rules["core"]:
                if word in text:
                    scores[capacity] += 2 * rules["weight"]
                    reasons.append(f"核心词'{word}'→{capacity}")

            # 机构特征匹配
            for word in rules["org"]:
                if word in text:
                    scores[capacity] += 1.5 * rules["weight"]

            # 活动特征匹配
            for word in rules["activity"]:
                if word in text:
                    scores[capacity] += 1 * rules["weight"]

        # 3. 计算最终分类
        total = sum(scores.values())
        if total == 0:
            return "协调能力", 0.33, "no_match"

        max_capacity = max(scores, key=scores.get)
        confidence = scores[max_capacity] / total

        return max_capacity, confidence, "; ".join(reasons[:3])

    def classify_batch(self, texts):
        """批量分类"""
        results = []
        for text in texts:
            label, conf, reason = self.classify(text)
            results.append({
                "text": text,
                "label": label,
                "confidence": conf,
                "reason": reason
            })
        return results


# ============================================================
# 真实LLM分类器（需要API密钥）
# ============================================================

class RealLLMClassifier:
    """
    真实LLM分类器
    支持OpenAI、Claude等API
    """

    def __init__(self, api_type="openai", api_key=None):
        self.api_type = api_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.prompt_template = CLASSIFICATION_PROMPT

    def classify(self, text):
        """
        使用真实LLM进行分类
        """
        if not self.api_key:
            raise ValueError("需要设置API密钥")

        prompt = self.prompt_template.format(contract_name=text)

        if self.api_type == "openai":
            return self._call_openai(prompt)
        elif self.api_type == "claude":
            return self._call_claude(prompt)
        else:
            raise ValueError(f"不支持的API类型: {self.api_type}")

    def _call_openai(self, prompt):
        """调用OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            result = response.choices[0].message.content.strip()

            # 解析结果
            for label in ["汲取能力", "协调能力", "合规能力"]:
                if label in result:
                    return label, 0.9, "llm"
            return "协调能力", 0.5, "llm_uncertain"
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return None, 0, str(e)

    def _call_claude(self, prompt):
        """调用Claude API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()

            for label in ["汲取能力", "协调能力", "合规能力"]:
                if label in result:
                    return label, 0.9, "llm"
            return "协调能力", 0.5, "llm_uncertain"
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            return None, 0, str(e)


# ============================================================
# 方案对比测试
# ============================================================

def compare_approaches():
    """对比不同分类方案"""

    print("=" * 80)
    print("国家能力合同分类 - LLM vs 关键词方法对比")
    print("=" * 80)

    # 加载数据
    df = pd.read_excel('/home/user/MOR-SI/2015.xls')
    print(f"\n数据集大小: {len(df)} 条")

    # 抽取测试样本
    test_samples = df['合同名称'].dropna().sample(100, random_state=42).tolist()

    print("\n" + "=" * 80)
    print("方案1: 模拟LLM分类 (增强语义规则)")
    print("=" * 80)

    llm_classifier = SimulatedLLMClassifier()
    llm_results = llm_classifier.classify_batch(test_samples)

    # 统计
    llm_labels = [r["label"] for r in llm_results]
    print(f"\n分类分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = llm_labels.count(label)
        print(f"  {label}: {count} ({count/len(llm_labels)*100:.1f}%)")

    # 显示样本
    print("\n样本分类结果:")
    for i, r in enumerate(llm_results[:10]):
        print(f"  {i+1}. {r['text'][:40]}...")
        print(f"     → {r['label']} (置信度: {r['confidence']:.2f})")
        print(f"     理由: {r['reason']}")

    # 与关键词方法对比
    print("\n" + "=" * 80)
    print("方案2: 原始关键词规则方法")
    print("=" * 80)

    from state_capacity_classifier import RuleBasedLabeler
    keyword_labeler = RuleBasedLabeler()

    keyword_results = []
    for text in test_samples:
        label, conf, keywords = keyword_labeler.label_contract(str(text) if text else "")
        keyword_results.append({
            "predicted_label": label,
            "confidence": conf,
            "matched_keywords": keywords
        })

    keyword_labels = [r["predicted_label"] for r in keyword_results]
    print(f"\n分类分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = keyword_labels.count(label)
        print(f"  {label}: {count} ({count/len(keyword_labels)*100:.1f}%)")

    # 对比一致性
    print("\n" + "=" * 80)
    print("方案对比")
    print("=" * 80)

    agreement = sum(1 for l, k in zip(llm_labels, keyword_labels) if l == k)
    print(f"\n两种方法一致性: {agreement}/{len(test_samples)} ({agreement/len(test_samples)*100:.1f}%)")

    # 分歧样本
    print("\n分歧样本分析:")
    disagreements = []
    for i, (text, llm_r, kw_r) in enumerate(zip(test_samples, llm_results, keyword_results)):
        if llm_r["label"] != kw_r["predicted_label"]:
            disagreements.append({
                "text": text,
                "llm_label": llm_r["label"],
                "llm_reason": llm_r["reason"],
                "keyword_label": kw_r["predicted_label"],
                "keywords": kw_r["matched_keywords"]
            })

    print(f"\n分歧数量: {len(disagreements)}")
    for i, d in enumerate(disagreements[:5]):
        print(f"\n  {i+1}. {d['text'][:50]}...")
        print(f"     LLM判断: {d['llm_label']} (理由: {d['llm_reason']})")
        print(f"     关键词判断: {d['keyword_label']} (关键词: {d['keywords']})")

    return llm_results, keyword_results, disagreements


def create_llm_training_set():
    """
    使用模拟LLM创建更高质量的训练集
    """
    print("\n" + "=" * 80)
    print("使用LLM创建训练集")
    print("=" * 80)

    df = pd.read_excel('/home/user/MOR-SI/2015.xls')
    llm_classifier = SimulatedLLMClassifier()

    # 对所有数据进行分类
    print(f"\n正在对 {len(df)} 条数据进行LLM分类...")

    results = []
    for idx, row in df.iterrows():
        text = row['合同名称']
        label, conf, reason = llm_classifier.classify(text)
        results.append({
            "text": text,
            "label": label,
            "confidence": conf
        })

        if (idx + 1) % 5000 == 0:
            print(f"  已处理: {idx + 1}/{len(df)}")

    # 筛选高置信度样本
    high_conf = [r for r in results if r["confidence"] >= 0.6]
    print(f"\n高置信度样本: {len(high_conf)} ({len(high_conf)/len(results)*100:.1f}%)")

    # 训练ML模型
    print("\n训练SVM模型...")

    label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
    texts = [r["text"] for r in high_conf if r["text"]]
    labels = [label_map[r["label"]] for r in high_conf if r["text"]]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def simple_tokenizer(text):
        if not text or pd.isna(text):
            return []
        text = str(text)
        return list(text) + [text[i:i+2] for i in range(len(text)-1)]

    vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC(max_iter=10000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\n模型评估:")
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1分数: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=["汲取能力", "协调能力", "合规能力"]))

    return results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 方案对比
    llm_results, keyword_results, disagreements = compare_approaches()

    # LLM训练集
    print("\n\n")
    llm_training_results = create_llm_training_set()
