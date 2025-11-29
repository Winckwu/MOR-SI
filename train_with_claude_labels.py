#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Claude语义标注的数据训练ML模型
并与之前的规则标注方法进行对比
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def simple_tokenizer(text):
    """中文分词器"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


def train_and_evaluate(texts, labels, label_source="Unknown"):
    """训练和评估模型"""
    print(f"\n{'='*60}")
    print(f"训练数据来源: {label_source}")
    print(f"{'='*60}")

    label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
    label_map_reverse = {0: "汲取能力", 1: "协调能力", 2: "合规能力"}

    # 转换标签
    y = [label_map[l] for l in labels]

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenizer,
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 训练多个模型
    models = {
        "SVM": LinearSVC(max_iter=10000, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "model": model,
            "y_pred": y_pred
        }
        print(f"  {name}: 准确率={acc:.4f}, F1={f1:.4f}")

    # 最佳模型
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_result = results[best_name]

    print(f"\n最佳模型: {best_name}")
    print("\n分类报告:")
    print(classification_report(
        y_test, best_result['y_pred'],
        target_names=["汲取能力", "协调能力", "合规能力"]
    ))

    return {
        "vectorizer": vectorizer,
        "best_model": best_result['model'],
        "best_name": best_name,
        "accuracy": best_result['accuracy'],
        "f1": best_result['f1'],
        "y_test": y_test,
        "y_pred": best_result['y_pred'],
        "all_results": results
    }


def main():
    print("="*70)
    print("对比分析：Claude语义标注 vs 规则标注")
    print("="*70)

    # ============================================
    # 1. 加载Claude语义标注数据
    # ============================================
    print("\n加载Claude语义标注数据...")
    with open('/home/user/MOR-SI/claude_semantic_labels_10000.json', 'r', encoding='utf-8') as f:
        claude_data = json.load(f)

    claude_labels = claude_data['labels']
    claude_texts = [r['合同名称'] for r in claude_labels]
    claude_tags = [r['label'] for r in claude_labels]

    print(f"Claude标注样本数: {len(claude_labels)}")
    print("Claude标注分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = claude_tags.count(label)
        print(f"  {label}: {count} ({count/len(claude_tags)*100:.1f}%)")

    # ============================================
    # 2. 使用规则标注同样的数据（对照组）
    # ============================================
    print("\n使用规则标注同样的数据...")

    # 导入规则标注器
    from enhanced_classifier import EnhancedRuleLabeler

    rule_labeler = EnhancedRuleLabeler()
    rule_results = rule_labeler.label_batch(claude_texts, show_progress=False)
    rule_tags = [r['label'] for r in rule_results]

    print("规则标注分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = rule_tags.count(label)
        print(f"  {label}: {count} ({count/len(rule_tags)*100:.1f}%)")

    # ============================================
    # 3. 分析两种标注的一致性
    # ============================================
    print("\n" + "="*70)
    print("两种标注方法的一致性分析")
    print("="*70)

    consistent = sum(1 for c, r in zip(claude_tags, rule_tags) if c == r)
    print(f"一致率: {consistent}/{len(claude_tags)} ({consistent/len(claude_tags)*100:.1f}%)")

    # 分歧统计
    disagreements = {}
    for c, r in zip(claude_tags, rule_tags):
        if c != r:
            key = f"{r}→{c}"
            disagreements[key] = disagreements.get(key, 0) + 1

    print("\n分歧类型统计 (规则标注→Claude标注):")
    for key, count in sorted(disagreements.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    # ============================================
    # 4. 训练模型对比
    # ============================================
    print("\n" + "="*70)
    print("模型训练对比")
    print("="*70)

    # 使用Claude标注训练
    claude_model_result = train_and_evaluate(
        claude_texts, claude_tags,
        "Claude语义理解标注"
    )

    # 使用规则标注训练
    rule_model_result = train_and_evaluate(
        claude_texts, rule_tags,
        "规则关键词标注"
    )

    # ============================================
    # 5. 交叉验证
    # ============================================
    print("\n" + "="*70)
    print("交叉验证对比 (重要！)")
    print("="*70)

    print("\n用Claude标注训练的模型，在规则标注的测试集上评估:")
    # 这能揭示两种标注的差异
    label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}

    X_all = claude_model_result['vectorizer'].transform(claude_texts)
    y_claude = [label_map[l] for l in claude_tags]
    y_rule = [label_map[l] for l in rule_tags]

    y_pred_by_claude_model = claude_model_result['best_model'].predict(X_all)

    # Claude模型在Claude标签上的准确率
    acc_claude_on_claude = accuracy_score(y_claude, y_pred_by_claude_model)
    # Claude模型在规则标签上的准确率
    acc_claude_on_rule = accuracy_score(y_rule, y_pred_by_claude_model)

    print(f"  Claude模型 vs Claude标签: {acc_claude_on_claude*100:.1f}%")
    print(f"  Claude模型 vs 规则标签: {acc_claude_on_rule*100:.1f}%")
    print(f"  差异: {(acc_claude_on_claude - acc_claude_on_rule)*100:.1f}个百分点")

    # ============================================
    # 6. 保存对比结果
    # ============================================
    print("\n" + "="*70)
    print("保存结果")
    print("="*70)

    # 保存Claude标注训练的模型
    model_data = {
        "vectorizer": claude_model_result['vectorizer'],
        "model": claude_model_result['best_model'],
        "model_name": claude_model_result['best_name'],
        "label_map": label_map,
        "label_map_reverse": {0: "汲取能力", 1: "协调能力", 2: "合规能力"},
        "accuracy": claude_model_result['accuracy'],
        "f1": claude_model_result['f1'],
        "label_source": "Claude语义理解标注",
        "created_at": datetime.now().isoformat()
    }

    with open('/home/user/MOR-SI/model_claude_semantic.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("模型已保存: model_claude_semantic.pkl")

    # 保存对比报告
    report = {
        "comparison": {
            "claude_labeling": {
                "distribution": {
                    "汲取能力": claude_tags.count("汲取能力"),
                    "协调能力": claude_tags.count("协调能力"),
                    "合规能力": claude_tags.count("合规能力")
                },
                "model_accuracy": claude_model_result['accuracy'],
                "model_f1": claude_model_result['f1']
            },
            "rule_labeling": {
                "distribution": {
                    "汲取能力": rule_tags.count("汲取能力"),
                    "协调能力": rule_tags.count("协调能力"),
                    "合规能力": rule_tags.count("合规能力")
                },
                "model_accuracy": rule_model_result['accuracy'],
                "model_f1": rule_model_result['f1']
            },
            "consistency_rate": consistent / len(claude_tags),
            "disagreements": disagreements,
            "cross_validation": {
                "claude_model_on_claude_labels": acc_claude_on_claude,
                "claude_model_on_rule_labels": acc_claude_on_rule
            }
        },
        "timestamp": datetime.now().isoformat()
    }

    with open('/home/user/MOR-SI/labeling_comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("对比报告已保存: labeling_comparison_report.json")

    # ============================================
    # 7. 总结
    # ============================================
    print("\n" + "="*70)
    print("关键发现总结")
    print("="*70)

    print(f"""
1. 标注一致性: {consistent/len(claude_tags)*100:.1f}%
   - 两种标注方法有 {100-consistent/len(claude_tags)*100:.1f}% 的分歧

2. 模型性能对比:
   - Claude标注训练: 准确率={claude_model_result['accuracy']*100:.1f}%, F1={claude_model_result['f1']:.4f}
   - 规则标注训练: 准确率={rule_model_result['accuracy']*100:.1f}%, F1={rule_model_result['f1']:.4f}

3. 关键洞察:
   - 如果两种标注一致性很高，说明Claude标注也是基于相似规则
   - 如果一致性低，说明Claude有不同的理解
   - 模型准确率高只说明能学会标注规则，不代表分类"正确"

4. 真正的验证需要:
   - 人工抽样验证（随机抽取200-500条人工标注）
   - 领域专家审核
""")

    return claude_model_result, rule_model_result, report


if __name__ == "__main__":
    main()
