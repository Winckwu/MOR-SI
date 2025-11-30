#!/usr/bin/env python3
"""
高级类别不平衡处理方法：
1. Prior概率调整 (阈值优化)
2. 代价敏感学习
3. 级联分类器
4. 基于规则的预分类 + ML
"""

import pandas as pd
import numpy as np
import glob
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data():
    """加载数据"""
    print("=" * 70)
    print("加载训练数据")
    print("=" * 70)

    all_files = glob.glob('classification_batch*.csv')
    dfs = []
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            dfs.append(df)
        except:
            pass

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['序号'])
    df = df[df['分类'] != '其他']

    df['text'] = df['采购人'].fillna('') + ' ' + df['合同名称'].fillna('')

    print(f"总记录数: {len(df)}")
    print(f"\n类别分布:")
    for cls, count in df['分类'].value_counts().items():
        print(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

    return df


def extract_rule_features(text):
    """基于规则提取特征 - 识别关键机构类型"""
    text = str(text)

    # 提取能力关键词 (税务、海关、统计等信息采集机构)
    extraction_keywords = [
        '税务', '国税', '地税', '海关', '统计局', '调查队',
        '测绘', '地质', '人民银行', '外汇', '国土资源',
        '统计', '普查'
    ]

    # 合规能力关键词 (公检法、监管执法机构)
    compliance_keywords = [
        '公安', '检察', '法院', '监狱', '司法', '纪委', '监察',
        '城管', '执法', '监督', '稽查', '缉私', '武警', '消防',
        '安监', '质监', '食药监', '市场监管', '检验检疫',
        '出入境', '边检', '海事'
    ]

    # 协调能力关键词 (教育、医疗、基础设施等公共服务)
    coordination_keywords = [
        '学校', '大学', '学院', '中学', '小学', '幼儿园', '教育',
        '医院', '卫生', '疾控', '农业', '林业', '水利', '交通',
        '公路', '铁路', '民航', '气象', '环保', '科技', '文化',
        '体育', '旅游', '民政', '人社', '住建'
    ]

    features = {
        'has_extraction': any(kw in text for kw in extraction_keywords),
        'has_compliance': any(kw in text for kw in compliance_keywords),
        'has_coordination': any(kw in text for kw in coordination_keywords),
    }

    return features


def rule_based_preclassify(texts):
    """基于规则的预分类"""
    predictions = []
    confidences = []

    for text in texts:
        features = extract_rule_features(text)

        # 计算每个类别的得分
        scores = {
            '提取能力': 2 if features['has_extraction'] else 0,
            '合规能力': 2 if features['has_compliance'] else 0,
            '协调能力': 1 if features['has_coordination'] else 0,  # 默认倾向
        }

        # 如果有多个匹配，使用优先级
        if features['has_extraction'] and not features['has_compliance']:
            pred = '提取能力'
            conf = 0.8
        elif features['has_compliance'] and not features['has_extraction']:
            pred = '合规能力'
            conf = 0.8
        elif features['has_extraction'] and features['has_compliance']:
            # 优先合规能力（如海关缉私）
            pred = '合规能力'
            conf = 0.6
        elif features['has_coordination']:
            pred = '协调能力'
            conf = 0.7
        else:
            pred = None  # 需要ML分类
            conf = 0.0

        predictions.append(pred)
        confidences.append(conf)

    return predictions, confidences


class PriorAdjustedClassifier:
    """Prior概率调整分类器"""

    def __init__(self, base_estimator, prior_weight=1.0):
        self.base_estimator = base_estimator
        self.prior_weight = prior_weight
        self.class_priors_ = None
        self.classes_ = None

    def fit(self, X, y):
        # 计算类别先验概率
        class_counts = Counter(y)
        total = sum(class_counts.values())
        self.classes_ = np.array(sorted(class_counts.keys()))
        self.class_priors_ = np.array([class_counts[c]/total for c in self.classes_])

        # 训练基础分类器
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        # 获取基础预测概率
        if hasattr(self.base_estimator, 'predict_proba'):
            proba = self.base_estimator.predict_proba(X)
        else:
            # 对于SVM等，使用decision_function
            decision = self.base_estimator.decision_function(X)
            if len(decision.shape) == 1:
                proba = np.column_stack([1-decision, decision])
            else:
                # 多分类：softmax
                exp_dec = np.exp(decision - decision.max(axis=1, keepdims=True))
                proba = exp_dec / exp_dec.sum(axis=1, keepdims=True)

        # 调整先验概率（贝叶斯后验调整）
        # P(y|x) ∝ P(x|y) * P(y)
        # 这里我们增加少数类的权重
        inverse_priors = 1.0 / (self.class_priors_ + 1e-10)
        inverse_priors = inverse_priors / inverse_priors.sum()

        adjusted_proba = proba * (inverse_priors ** self.prior_weight)
        adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)

        return adjusted_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ThresholdOptimizedClassifier:
    """阈值优化分类器 - 针对每个类别优化决策阈值"""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.thresholds_ = None
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_

        # 如果没有验证集，使用训练集的一部分
        if X_val is None:
            X_val, y_val = X, y

        # 获取验证集预测概率
        if hasattr(self.base_estimator, 'predict_proba'):
            proba = self.base_estimator.predict_proba(X_val)
        else:
            decision = self.base_estimator.decision_function(X_val)
            exp_dec = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = exp_dec / exp_dec.sum(axis=1, keepdims=True)

        # 为每个类别找最优阈值
        self.thresholds_ = np.ones(len(self.classes_)) * 0.5

        for i, cls in enumerate(self.classes_):
            best_f1 = 0
            best_thresh = 0.5

            for thresh in np.arange(0.1, 0.9, 0.05):
                y_pred_binary = (proba[:, i] >= thresh).astype(int)
                y_true_binary = (y_val == cls).astype(int)

                if y_pred_binary.sum() > 0:
                    f1 = f1_score(y_true_binary, y_pred_binary)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh

            self.thresholds_[i] = best_thresh

        print(f"优化后的阈值: {dict(zip(self.classes_, self.thresholds_))}")
        return self

    def predict(self, X):
        if hasattr(self.base_estimator, 'predict_proba'):
            proba = self.base_estimator.predict_proba(X)
        else:
            decision = self.base_estimator.decision_function(X)
            exp_dec = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = exp_dec / exp_dec.sum(axis=1, keepdims=True)

        # 使用优化后的阈值
        adjusted_scores = proba / self.thresholds_
        return self.classes_[np.argmax(adjusted_scores, axis=1)]


class CascadeClassifier:
    """级联分类器 - 先识别少数类，再分类多数类"""

    def __init__(self):
        self.stage1_model = None  # 二分类：是否为少数类
        self.stage2_model = None  # 少数类内部分类
        self.stage3_model = None  # 多数类分类
        self.minority_classes = None
        self.le = None

    def fit(self, X, y, minority_classes=['合规能力', '提取能力']):
        self.minority_classes = minority_classes
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)

        # Stage 1: 二分类 - 是否为少数类
        y_binary = np.isin(y, minority_classes).astype(int)
        self.stage1_model = CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', random_state=RANDOM_STATE),
            cv=3
        )
        self.stage1_model.fit(X, y_binary)

        # Stage 2: 少数类内部分类
        minority_mask = np.isin(y, minority_classes)
        if minority_mask.sum() > 0:
            X_minority = X[minority_mask]
            y_minority = y[minority_mask]
            self.stage2_model = LogisticRegression(
                class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000
            )
            self.stage2_model.fit(X_minority, y_minority)

        # Stage 3: 如果不是少数类，默认为协调能力
        # (在这个数据集中，非少数类只有"协调能力")

        return self

    def predict(self, X):
        predictions = []

        # Stage 1: 预测是否为少数类
        proba_minority = self.stage1_model.predict_proba(X)[:, 1]

        for i in range(X.shape[0]):
            if proba_minority[i] > 0.3:  # 降低阈值，更容易识别少数类
                # Stage 2: 少数类内部分类
                pred = self.stage2_model.predict(X[i:i+1])[0]
            else:
                # Stage 3: 多数类
                pred = '协调能力'
            predictions.append(pred)

        return np.array(predictions)


class HybridRuleMLClassifier:
    """混合分类器：规则 + 机器学习"""

    def __init__(self, ml_model):
        self.ml_model = ml_model
        self.le = None

    def fit(self, X, y, texts):
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.ml_model.fit(X, y_encoded)
        return self

    def predict(self, X, texts):
        # 首先使用规则分类
        rule_preds, rule_confs = rule_based_preclassify(texts)

        # ML预测
        ml_preds_encoded = self.ml_model.predict(X)
        ml_preds = self.le.inverse_transform(ml_preds_encoded)

        # 混合：高置信度使用规则，低置信度使用ML
        final_preds = []
        for rule_pred, rule_conf, ml_pred in zip(rule_preds, rule_confs, ml_preds):
            if rule_pred is not None and rule_conf >= 0.7:
                final_preds.append(rule_pred)
            else:
                final_preds.append(ml_pred)

        return np.array(final_preds)


def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"\n{model_name}:")
    print(f"  准确率: {acc:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")

    # 每个类别的召回率
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(f"  各类别召回率:")
    for cls in ['协调能力', '合规能力', '提取能力']:
        if cls in report:
            print(f"    {cls}: {report[cls]['recall']:.4f}")

    return acc, f1_weighted, f1_macro


def main():
    print("\n" + "=" * 70)
    print("高级类别不平衡处理方法对比")
    print("=" * 70)

    # 加载数据
    df = load_data()

    # 准备特征
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['分类'])

    X = df['text']
    y = df['分类'].values

    # 划分数据
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, df['text'].values, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 向量化
    vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 3), max_features=15000, min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"\n训练集: {X_train_vec.shape[0]}, 测试集: {X_test_vec.shape[0]}")

    results = []

    # 1. 基线模型
    print("\n" + "=" * 70)
    print("1. 基线模型 (Linear SVM)")
    print("=" * 70)

    baseline = LinearSVC(random_state=RANDOM_STATE, max_iter=3000)
    baseline.fit(X_train_vec, y_train)
    y_pred = baseline.predict(X_test_vec)
    acc, f1_w, f1_m = evaluate_model(y_test, y_pred, "基线 LinearSVC")
    results.append(('基线 LinearSVC', acc, f1_w, f1_m))

    # 2. Prior概率调整
    print("\n" + "=" * 70)
    print("2. Prior概率调整分类器")
    print("=" * 70)

    for prior_weight in [0.5, 1.0, 1.5, 2.0]:
        prior_clf = PriorAdjustedClassifier(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            prior_weight=prior_weight
        )
        prior_clf.fit(X_train_vec, y_train)
        y_pred = prior_clf.predict(X_test_vec)
        acc, f1_w, f1_m = evaluate_model(y_test, y_pred, f"Prior调整 (weight={prior_weight})")
        results.append((f'Prior调整 (w={prior_weight})', acc, f1_w, f1_m))

    # 3. 阈值优化
    print("\n" + "=" * 70)
    print("3. 阈值优化分类器")
    print("=" * 70)

    threshold_clf = ThresholdOptimizedClassifier(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    threshold_clf.fit(X_train_vec, y_train)
    y_pred = threshold_clf.predict(X_test_vec)
    acc, f1_w, f1_m = evaluate_model(y_test, y_pred, "阈值优化")
    results.append(('阈值优化', acc, f1_w, f1_m))

    # 4. 级联分类器
    print("\n" + "=" * 70)
    print("4. 级联分类器")
    print("=" * 70)

    cascade_clf = CascadeClassifier()
    cascade_clf.fit(X_train_vec, y_train)
    y_pred = cascade_clf.predict(X_test_vec)
    acc, f1_w, f1_m = evaluate_model(y_test, y_pred, "级联分类器")
    results.append(('级联分类器', acc, f1_w, f1_m))

    # 5. 混合规则+ML
    print("\n" + "=" * 70)
    print("5. 混合分类器 (规则 + ML)")
    print("=" * 70)

    hybrid_clf = HybridRuleMLClassifier(
        LinearSVC(random_state=RANDOM_STATE, max_iter=3000)
    )
    hybrid_clf.fit(X_train_vec, y_train, texts_train)
    y_pred = hybrid_clf.predict(X_test_vec, texts_test)
    acc, f1_w, f1_m = evaluate_model(y_test, y_pred, "混合分类器")
    results.append(('混合 (规则+ML)', acc, f1_w, f1_m))

    # 6. 代价敏感 + 集成
    print("\n" + "=" * 70)
    print("6. 代价敏感集成分类器")
    print("=" * 70)

    # 使用 'balanced' 自动计算权重
    ensemble = VotingClassifier(
        estimators=[
            ('svm', LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=3000)),
            ('lr', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)),
            ('rf', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=RANDOM_STATE))
        ],
        voting='hard'
    )
    ensemble.fit(X_train_vec, y_train)
    y_pred = ensemble.predict(X_test_vec)
    acc, f1_w, f1_m = evaluate_model(y_test, y_pred, "代价敏感集成")
    results.append(('代价敏感集成', acc, f1_w, f1_m))

    # 结果汇总
    print("\n" + "=" * 70)
    print("结果汇总 (按Macro F1排序)")
    print("=" * 70)

    results_df = pd.DataFrame(results, columns=['模型', '准确率', 'F1_weighted', 'F1_macro'])
    results_df = results_df.sort_values('F1_macro', ascending=False)

    print(f"\n{'排名':<4} {'模型':<25} {'准确率':<10} {'F1_weighted':<12} {'F1_macro':<10}")
    print("-" * 65)
    for i, row in results_df.iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"{rank:<4} {row['模型']:<25} {row['准确率']:.4f}     {row['F1_weighted']:.4f}       {row['F1_macro']:.4f}")

    # 最佳模型详细报告
    best_model = results_df.iloc[0]['模型']
    print(f"\n最佳模型: {best_model}")
    print(f"Macro F1提升: {(results_df.iloc[0]['F1_macro'] - results[0][3]) * 100:.2f}%")

    # 保存结果
    results_df.to_csv('imbalance_comparison_results.csv', index=False)
    print(f"\n结果已保存到 imbalance_comparison_results.csv")


if __name__ == '__main__':
    main()
