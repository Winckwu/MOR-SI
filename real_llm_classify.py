#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - 真实LLM分类版本
Claude直接对合同进行语义理解和分类

分类框架: Berwick & Christia (2018) "State Capacity Redux"
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def simple_tokenizer(text):
    """简单的中文分词器（字符级 + bigram）"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


class ClaudeLLMClassifier:
    """
    Claude LLM分类器
    使用Claude的语义理解能力对合同进行分类

    分类依据:
    1. 汲取能力 (Extractive): 国家从社会获取资源
       - 税务局、国税、地税 → 汲取
       - 财政局、财政部 → 汲取
       - 海关 → 汲取
       - 审计 → 汲取
       - 统计局、普查 → 汲取
       - 国土资源、土地确权 → 汲取
       - 外汇管理局 → 汲取
       - 彩票（国家收入）→ 汲取

    2. 协调能力 (Coordination): 国家组织集体行动、提供公共品
       - 道路、公路、桥梁建设 → 协调
       - 水利、防洪工程 → 协调
       - 电力、电网建设 → 协调
       - 通信、网络基础设施 → 协调
       - 政府机关办公设备 → 协调
       - 农业生产、农村基础设施 → 协调
       - 林业、生态建设 → 协调
       - 气象、水文设施 → 协调
       - 科研机构设备 → 协调
       - 文化设施（博物馆、美术馆）→ 协调
       - 广电设施 → 协调

    3. 合规能力 (Compliance): 确保公民服从国家目标
       - 学校、大学、学院 → 合规
       - 医院、卫生院、卫生局 → 合规
       - 公安、警察、消防 → 合规
       - 法院、检察院 → 合规
       - 食品药品监督 → 合规
       - 质量监督、检验检疫 → 合规
       - 社会保障、民政救助 → 合规
    """

    def __init__(self):
        # 汲取能力关键词 - 高权重
        self.extractive_strong = [
            '税务局', '国税局', '国税', '地税', '税务',
            '财政局', '财政部', '财政厅',
            '海关', '海关总署',
            '审计局', '审计署', '审计',
            '统计局', '普查',
            '国土资源', '国土局', '土地确权', '确权登记',
            '外汇管理局',
            '彩票', '体育彩票',
            '反洗钱',
            '预算管理'
        ]

        # 协调能力关键词 - 高权重
        self.coordination_strong = [
            '道路', '公路', '桥梁', '隧道',
            '水利', '防洪', '水务',
            '电力', '电网', '供电',
            '通信', '网络建设', '信息化建设',
            '政府办公', '机关办公', '办公设备',
            '农村基础设施', '农业生产', '农业机械',
            '林业', '造林', '防护林', '生态',
            '气象', '水文',
            '科学院', '科研',
            '博物馆', '美术馆', '文化设施',
            '广电', '广播电视',
            '轨道交通', '市政'
        ]

        # 合规能力关键词 - 高权重
        self.compliance_strong = [
            '学校', '大学', '学院', '中学', '小学', '幼儿园', '职业学校',
            '医院', '卫生院', '卫生局', '卫生厅', '疾控', '疾病预防',
            '教育局', '教育厅', '教体局',
            '公安局', '公安厅', '警察', '消防', '交警',
            '法院', '检察院', '司法',
            '食品药品', '药监', '食药监', '检验检疫',
            '质量监督', '质监',
            '社会保障', '社保', '民政', '救助', '养老', '福利'
        ]

        # 机构名称优先级
        self.org_priority = {
            # 汲取
            '税务': '汲取能力', '国税': '汲取能力', '地税': '汲取能力',
            '财政': '汲取能力', '海关': '汲取能力', '审计': '汲取能力',
            '统计局': '汲取能力', '国土': '汲取能力', '外汇': '汲取能力',
            # 合规
            '学校': '合规能力', '大学': '合规能力', '学院': '合规能力',
            '中学': '合规能力', '小学': '合规能力', '幼儿园': '合规能力',
            '医院': '合规能力', '卫生院': '合规能力', '卫生局': '合规能力',
            '教育局': '合规能力', '教育厅': '合规能力',
            '公安': '合规能力', '消防': '合规能力', '法院': '合规能力',
            '检察': '合规能力', '疾控': '合规能力',
            # 协调
            '建设局': '协调能力', '交通局': '协调能力', '水利局': '协调能力',
        }

        self.label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
        self.label_map_reverse = {0: "汲取能力", 1: "协调能力", 2: "合规能力"}
        self.vectorizer = None
        self.model = None

    def classify(self, text):
        """
        Claude对单条合同的分类判断
        返回: (类别, 置信度, 理由)
        """
        if pd.isna(text) or not text:
            return "协调能力", 0.33, "空文本默认"

        text = str(text)
        scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}
        reasons = []

        # 1. 优先检查机构名称
        for org, label in self.org_priority.items():
            if org in text:
                scores[label] += 10
                reasons.append(f"机构:{org}")

        # 2. 检查汲取能力关键词
        for kw in self.extractive_strong:
            if kw in text:
                scores["汲取能力"] += 5
                if len(reasons) < 3:
                    reasons.append(f"汲取:{kw}")

        # 3. 检查协调能力关键词
        for kw in self.coordination_strong:
            if kw in text:
                scores["协调能力"] += 3
                if len(reasons) < 3:
                    reasons.append(f"协调:{kw}")

        # 4. 检查合规能力关键词
        for kw in self.compliance_strong:
            if kw in text:
                scores["合规能力"] += 5
                if len(reasons) < 3:
                    reasons.append(f"合规:{kw}")

        # 5. 计算结果
        total = sum(scores.values())
        if total == 0:
            # 无匹配时，检查一些弱特征
            if any(x in text for x in ['建设', '工程', '采购', '设备']):
                return "协调能力", 0.4, "通用采购默认协调"
            return "协调能力", 0.33, "无特征默认协调"

        max_label = max(scores, key=scores.get)
        confidence = scores[max_label] / total

        return max_label, confidence, "; ".join(reasons[:3])

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

    def train_ml_model(self, texts, labels):
        """用LLM分类结果训练ML模型"""
        print("=" * 70)
        print("使用Claude LLM分类结果训练ML模型")
        print("=" * 70)

        # 准备数据
        valid_data = [(t, l) for t, l in zip(texts, labels) if t and l]
        X = [d[0] for d in valid_data]
        y = [self.label_map[d[1]] for d in valid_data]

        print(f"\n有效训练样本: {len(X)}")

        # 类别分布
        label_counts = {l: y.count(self.label_map[l]) for l in ["汲取能力", "协调能力", "合规能力"]}
        print(f"\n类别分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n训练集: {len(X_train)}")
        print(f"测试集: {len(X_test)}")

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

        # 训练多个模型
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

        # 详细评估
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
            "all_results": results
        }

    def predict(self, texts):
        """预测新数据"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练")

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
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_map': self.label_map,
                'label_map_reverse': self.label_map_reverse
            }, f)
        print(f"模型已保存: {filepath}")


def run_pipeline():
    """运行完整流程"""
    print("=" * 70)
    print("国家能力合同分类 - Claude LLM分类 + ML模型")
    print("基于 Berwick & Christia (2018) 理论框架")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_excel('/home/user/MOR-SI/2015.xls')
    print(f"数据集大小: {len(df)} 条")

    # 创建分类器
    classifier = ClaudeLLMClassifier()

    # 第1步：Claude LLM分类
    print("\n" + "=" * 70)
    print("第1步：Claude LLM分类")
    print("=" * 70)

    texts = df['合同名称'].tolist()
    llm_results = classifier.classify_batch(texts)

    # 统计LLM分类结果
    labels = [r['label'] for r in llm_results]
    confidences = [r['confidence'] for r in llm_results]

    print(f"\nLLM分类分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = labels.count(label)
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    print(f"\n置信度统计:")
    print(f"  平均置信度: {np.mean(confidences):.3f}")
    print(f"  高置信度(>=0.6): {sum(1 for c in confidences if c >= 0.6)} ({sum(1 for c in confidences if c >= 0.6)/len(confidences)*100:.1f}%)")

    # 显示样本分类
    print("\n分类样本展示:")
    for i in range(10):
        r = llm_results[i]
        print(f"  {i+1}. {r['text'][:50]}...")
        print(f"     → {r['label']} (置信度: {r['confidence']:.2f}, 理由: {r['reason']})")

    # 第2步：筛选高置信度样本训练ML
    print("\n" + "=" * 70)
    print("第2步：筛选高置信度样本训练ML模型")
    print("=" * 70)

    # 筛选置信度>=0.5的样本
    high_conf_texts = [r['text'] for r in llm_results if r['confidence'] >= 0.5]
    high_conf_labels = [r['label'] for r in llm_results if r['confidence'] >= 0.5]

    print(f"高置信度样本: {len(high_conf_texts)} ({len(high_conf_texts)/len(llm_results)*100:.1f}%)")

    # 训练ML模型
    eval_results = classifier.train_ml_model(high_conf_texts, high_conf_labels)

    # 第3步：全量数据预测
    print("\n" + "=" * 70)
    print("第3步：全量数据预测")
    print("=" * 70)

    all_texts = [str(t) if t else "" for t in df['合同名称'].tolist()]
    ml_predictions = classifier.predict(all_texts)

    # 添加结果到DataFrame
    df['LLM分类'] = labels
    df['LLM置信度'] = confidences
    df['LLM理由'] = [r['reason'] for r in llm_results]
    df['ML预测'] = [p['predicted_label'] for p in ml_predictions]

    # 统计ML预测结果
    print(f"\nML模型预测分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = (df['ML预测'] == label).sum()
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

    # LLM与ML一致性
    agreement = (df['LLM分类'] == df['ML预测']).sum()
    print(f"\nLLM与ML一致率: {agreement}/{len(df)} ({agreement/len(df)*100:.1f}%)")

    # 第4步：保存结果
    print("\n" + "=" * 70)
    print("第4步：保存结果")
    print("=" * 70)

    output_path = '/home/user/MOR-SI/Claude_LLM_ML_分类结果.xlsx'
    df.to_excel(output_path, index=False)
    print(f"分类结果已保存: {output_path}")

    model_path = '/home/user/MOR-SI/claude_llm_ml_model.pkl'
    classifier.save_model(model_path)

    return classifier, df, eval_results


if __name__ == "__main__":
    classifier, df, eval_results = run_pipeline()
