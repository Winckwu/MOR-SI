#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练两个方案的ML模型并对比效果
生成可直接使用的模型文件
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from enhanced_classifier import (
    EnhancedRuleLabeler, EnhancedLLMLabeler, simple_tokenizer
)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
        self.label_map_reverse = {0: "汲取能力", 1: "协调能力", 2: "合规能力"}

    def train_model(self, texts, labels, model_name="SVM"):
        """训练单个模型"""
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(
            tokenizer=simple_tokenizer,
            max_features=5000,
            ngram_range=(1, 2)
        )

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # 选择模型
        models = {
            "SVM": LinearSVC(max_iter=10000, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        # 训练所有模型，选择最佳
        best_model = None
        best_score = 0
        best_name = ""
        results = {}

        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            results[name] = {"accuracy": acc, "f1": f1, "model": model}

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        # 获取最佳模型的详细评估
        y_pred = best_model.predict(X_test_tfidf)

        return {
            "vectorizer": vectorizer,
            "model": best_model,
            "best_model_name": best_name,
            "accuracy": results[best_name]["accuracy"],
            "f1": results[best_name]["f1"],
            "y_test": y_test,
            "y_pred": y_pred,
            "all_results": results,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }


def train_both_approaches(data_path, text_column="合同名称", output_dir="."):
    """训练两个方案的模型并对比"""

    print("=" * 70)
    print("国家能力合同分类器 - 双方案模型训练")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_excel(data_path)
    print(f"数据集大小: {len(df)} 条")

    texts = df[text_column].tolist()
    trainer = ModelTrainer()

    # ========================================
    # 方案1：增强关键词规则 + ML
    # ========================================
    print("\n" + "=" * 70)
    print("方案1：增强关键词规则 + ML模型")
    print("=" * 70)

    rule_labeler = EnhancedRuleLabeler()
    rule_results = rule_labeler.label_batch(texts, show_progress=True)

    # 筛选高置信度样本
    rule_texts = []
    rule_labels = []
    for r in rule_results:
        if r["confidence"] >= trainer.confidence_threshold and r["text"]:
            rule_texts.append(str(r["text"]))
            rule_labels.append(trainer.label_map[r["label"]])

    print(f"\n高置信度训练样本: {len(rule_texts)} ({len(rule_texts)/len(texts)*100:.1f}%)")

    # 类别分布
    print("\n训练集类别分布:")
    for label_name, label_id in trainer.label_map.items():
        count = rule_labels.count(label_id)
        print(f"  {label_name}: {count} ({count/len(rule_labels)*100:.1f}%)")

    # 训练模型
    print("\n训练模型...")
    rule_model_result = trainer.train_model(rule_texts, rule_labels)

    print(f"\n最佳模型: {rule_model_result['best_model_name']}")
    print(f"测试准确率: {rule_model_result['accuracy']:.4f}")
    print(f"测试F1分数: {rule_model_result['f1']:.4f}")

    print("\n分类报告:")
    print(classification_report(
        rule_model_result['y_test'],
        rule_model_result['y_pred'],
        target_names=["汲取能力", "协调能力", "合规能力"]
    ))

    # ========================================
    # 方案2：增强LLM语义规则 + ML
    # ========================================
    print("\n" + "=" * 70)
    print("方案2：增强LLM语义规则 + ML模型")
    print("=" * 70)

    llm_labeler = EnhancedLLMLabeler()
    llm_results = llm_labeler.label_batch(texts, show_progress=True)

    # 筛选高置信度样本
    llm_texts = []
    llm_labels = []
    for r in llm_results:
        if r["confidence"] >= trainer.confidence_threshold and r["text"]:
            llm_texts.append(str(r["text"]))
            llm_labels.append(trainer.label_map[r["label"]])

    print(f"\n高置信度训练样本: {len(llm_texts)} ({len(llm_texts)/len(texts)*100:.1f}%)")

    # 类别分布
    print("\n训练集类别分布:")
    for label_name, label_id in trainer.label_map.items():
        count = llm_labels.count(label_id)
        print(f"  {label_name}: {count} ({count/len(llm_labels)*100:.1f}%)")

    # 训练模型
    print("\n训练模型...")
    llm_model_result = trainer.train_model(llm_texts, llm_labels)

    print(f"\n最佳模型: {llm_model_result['best_model_name']}")
    print(f"测试准确率: {llm_model_result['accuracy']:.4f}")
    print(f"测试F1分数: {llm_model_result['f1']:.4f}")

    print("\n分类报告:")
    print(classification_report(
        llm_model_result['y_test'],
        llm_model_result['y_pred'],
        target_names=["汲取能力", "协调能力", "合规能力"]
    ))

    # ========================================
    # 对比两个方案
    # ========================================
    print("\n" + "=" * 70)
    print("两个方案模型对比")
    print("=" * 70)

    print("\n| 指标 | 方案1(关键词规则) | 方案2(LLM语义规则) |")
    print("|------|-------------------|-------------------|")
    print(f"| 训练样本数 | {rule_model_result['train_size']} | {llm_model_result['train_size']} |")
    print(f"| 最佳模型 | {rule_model_result['best_model_name']} | {llm_model_result['best_model_name']} |")
    print(f"| 测试准确率 | {rule_model_result['accuracy']:.4f} | {llm_model_result['accuracy']:.4f} |")
    print(f"| 测试F1分数 | {rule_model_result['f1']:.4f} | {llm_model_result['f1']:.4f} |")

    # 使用两个模型对全量数据进行预测并对比
    print("\n" + "=" * 70)
    print("全量数据预测对比")
    print("=" * 70)

    all_texts = [str(t) if t else "" for t in texts]

    # 方案1预测
    X_all_rule = rule_model_result['vectorizer'].transform(all_texts)
    pred_rule = rule_model_result['model'].predict(X_all_rule)
    pred_rule_labels = [trainer.label_map_reverse[p] for p in pred_rule]

    # 方案2预测
    X_all_llm = llm_model_result['vectorizer'].transform(all_texts)
    pred_llm = llm_model_result['model'].predict(X_all_llm)
    pred_llm_labels = [trainer.label_map_reverse[p] for p in pred_llm]

    # 统计预测分布
    print("\n方案1 ML模型预测分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = pred_rule_labels.count(label)
        print(f"  {label}: {count} ({count/len(pred_rule_labels)*100:.1f}%)")

    print("\n方案2 ML模型预测分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = pred_llm_labels.count(label)
        print(f"  {label}: {count} ({count/len(pred_llm_labels)*100:.1f}%)")

    # 两个模型预测一致性
    consistent = sum(1 for r1, r2 in zip(pred_rule_labels, pred_llm_labels) if r1 == r2)
    print(f"\n两个模型预测一致率: {consistent/len(pred_rule_labels)*100:.1f}%")

    # ========================================
    # 保存模型
    # ========================================
    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)

    # 保存方案1模型
    model1_path = os.path.join(output_dir, "model_rule_based.pkl")
    model1_data = {
        "vectorizer": rule_model_result['vectorizer'],
        "model": rule_model_result['model'],
        "model_name": rule_model_result['best_model_name'],
        "label_map": trainer.label_map,
        "label_map_reverse": trainer.label_map_reverse,
        "accuracy": rule_model_result['accuracy'],
        "f1": rule_model_result['f1'],
        "approach": "方案1：增强关键词规则",
        "created_at": datetime.now().isoformat()
    }
    with open(model1_path, 'wb') as f:
        pickle.dump(model1_data, f)
    print(f"方案1模型已保存: {model1_path}")

    # 保存方案2模型
    model2_path = os.path.join(output_dir, "model_llm_semantic.pkl")
    model2_data = {
        "vectorizer": llm_model_result['vectorizer'],
        "model": llm_model_result['model'],
        "model_name": llm_model_result['best_model_name'],
        "label_map": trainer.label_map,
        "label_map_reverse": trainer.label_map_reverse,
        "accuracy": llm_model_result['accuracy'],
        "f1": llm_model_result['f1'],
        "approach": "方案2：增强LLM语义规则",
        "created_at": datetime.now().isoformat()
    }
    with open(model2_path, 'wb') as f:
        pickle.dump(model2_data, f)
    print(f"方案2模型已保存: {model2_path}")

    # 保存完整结果
    result_df = df.copy()
    result_df["方案1_规则标注"] = [r["label"] for r in rule_results]
    result_df["方案1_规则置信度"] = [r["confidence"] for r in rule_results]
    result_df["方案1_ML预测"] = pred_rule_labels
    result_df["方案2_规则标注"] = [r["label"] for r in llm_results]
    result_df["方案2_规则置信度"] = [r["confidence"] for r in llm_results]
    result_df["方案2_ML预测"] = pred_llm_labels
    result_df["两模型一致"] = [r1 == r2 for r1, r2 in zip(pred_rule_labels, pred_llm_labels)]

    result_path = os.path.join(output_dir, "双方案完整分类结果.xlsx")
    result_df.to_excel(result_path, index=False)
    print(f"完整结果已保存: {result_path}")

    return {
        "model1": model1_data,
        "model2": model2_data,
        "result_df": result_df
    }


if __name__ == "__main__":
    train_both_approaches("/home/user/MOR-SI/2015.xls", "合同名称", "/home/user/MOR-SI")
