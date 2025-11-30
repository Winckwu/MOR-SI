#!/usr/bin/env python3
"""
纯机器学习优化 - 不使用任何规则
专注于：
1. 更好的特征表示
2. 类别权重优化
3. 阈值调整
4. 模型集成
5. Focal Loss思想的实现
"""

import pandas as pd
import numpy as np
import glob
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    make_scorer
)

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data():
    """加载所有标注数据"""
    print("=" * 70)
    print("加载训练数据 (近2万条标注数据)")
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
    df = df[df['分类'] != '其他']  # 移除极少数的"其他"类

    df['text'] = df['采购人'].fillna('') + ' ' + df['合同名称'].fillna('')

    print(f"总标注数据: {len(df)} 条")
    print(f"\n类别分布:")
    for cls, count in df['分类'].value_counts().items():
        print(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

    return df


def create_vectorizers():
    """创建多种向量化器进行对比"""
    vectorizers = {
        'tfidf_word_1_2': TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), max_features=10000, min_df=2
        ),
        'tfidf_word_1_3': TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3), max_features=15000, min_df=2
        ),
        'tfidf_char_2_4': TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4), max_features=15000, min_df=2
        ),
        'tfidf_char_word_combined': TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5), max_features=15000, min_df=2
        ),
    }
    return vectorizers


class FocalLossLogisticRegression:
    """
    Focal Loss 思想的逻辑回归
    对难分类的样本给予更高权重
    """
    def __init__(self, gamma=2.0, n_iterations=3):
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # 初始权重
        sample_weights = np.ones(len(y))

        for iteration in range(self.n_iterations):
            # 训练模型
            self.model = LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                class_weight='balanced'
            )
            self.model.fit(X, y, sample_weight=sample_weights)

            # 获取预测概率
            proba = self.model.predict_proba(X)

            # 计算 focal weight: (1 - p_correct)^gamma
            # 对于每个样本，找到其真实类别的预测概率
            correct_proba = np.zeros(len(y))
            for i, (yi, pi) in enumerate(zip(y, proba)):
                class_idx = np.where(self.classes_ == yi)[0][0]
                correct_proba[i] = pi[class_idx]

            # Focal weight: 预测错误的概率越高，权重越大
            sample_weights = (1 - correct_proba) ** self.gamma
            sample_weights = sample_weights / sample_weights.mean()  # 归一化

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class AdaptiveThresholdClassifier:
    """
    自适应阈值分类器
    针对每个类别学习最优决策阈值
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.thresholds = None
        self.classes_ = None

    def fit(self, X, y):
        # 先用交叉验证找最优阈值
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        all_proba = np.zeros((len(y), len(np.unique(y))))
        all_true = np.zeros(len(y))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = CalibratedClassifierCV(
                self.base_model.__class__(**self.base_model.get_params()),
                cv=3
            )
            model.fit(X_tr, y_tr)

            all_proba[val_idx] = model.predict_proba(X_val)
            all_true[val_idx] = y_val

        self.classes_ = np.unique(y)

        # 为每个类找最优阈值 (最大化F1)
        self.thresholds = {}
        for i, cls in enumerate(self.classes_):
            best_f1 = 0
            best_thresh = 0.5

            for thresh in np.arange(0.1, 0.9, 0.02):
                y_pred_binary = (all_proba[:, i] >= thresh).astype(int)
                y_true_binary = (all_true == cls).astype(int)

                if y_pred_binary.sum() > 0:
                    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
                    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
                    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh

            self.thresholds[cls] = best_thresh

        print(f"优化阈值: {self.thresholds}")

        # 最后用全部数据训练
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)
        else:
            # 需要校准
            cal_model = CalibratedClassifierCV(self.base_model, cv='prefit')
            proba = cal_model.predict_proba(X)

        # 使用阈值调整后的分数
        adjusted_scores = np.zeros_like(proba)
        for i, cls in enumerate(self.classes_):
            adjusted_scores[:, i] = proba[:, i] / self.thresholds[cls]

        return self.classes_[np.argmax(adjusted_scores, axis=1)]


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, le):
    """训练并评估模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')

    # 各类别召回率 - 使用target_names让报告使用原始类名
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

    recalls = {}
    for cls in le.classes_:
        if cls in report:
            recalls[cls] = report[cls]['recall']

    return {
        'name': name,
        'accuracy': acc,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'recalls': recalls,
        'model': model
    }


def main():
    print("\n" + "=" * 70)
    print("纯机器学习优化 - 利用近2万条标注数据")
    print("=" * 70)

    # 加载数据
    df = load_data()

    le = LabelEncoder()
    y = le.fit_transform(df['分类'])
    X_text = df['text'].values

    # 划分数据
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n训练集: {len(X_train_text)}, 测试集: {len(X_test_text)}")

    # 计算类别权重
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    class_weight_dict = {i: total / (len(class_counts) * count)
                         for i, count in class_counts.items()}

    all_results = []

    # ============ 测试不同向量化方法 ============
    print("\n" + "=" * 70)
    print("1. 向量化方法对比")
    print("=" * 70)

    best_vectorizer = None
    best_vec_f1 = 0

    for vec_name, vectorizer in create_vectorizers().items():
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        model = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=3000)
        result = train_and_evaluate(
            f"SVM + {vec_name}", model, X_train, X_test, y_train, y_test, le
        )

        print(f"\n{vec_name}:")
        print(f"  准确率: {result['accuracy']:.4f}, F1_macro: {result['f1_macro']:.4f}")
        print(f"  召回率: {result['recalls']}")

        all_results.append(result)

        if result['f1_macro'] > best_vec_f1:
            best_vec_f1 = result['f1_macro']
            best_vectorizer = vectorizer
            best_vec_name = vec_name

    print(f"\n最佳向量化: {best_vec_name} (F1_macro={best_vec_f1:.4f})")

    # 使用最佳向量化
    X_train = best_vectorizer.fit_transform(X_train_text)
    X_test = best_vectorizer.transform(X_test_text)

    # ============ 测试不同模型 ============
    print("\n" + "=" * 70)
    print("2. 模型对比 (使用最佳向量化)")
    print("=" * 70)

    models = {
        'LinearSVC': LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=3000),

        'LogisticRegression': LogisticRegression(
            class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000, C=1.0
        ),

        'LogisticRegression_C10': LogisticRegression(
            class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000, C=10.0
        ),

        'SGD_log': SGDClassifier(
            loss='log_loss', class_weight='balanced', random_state=RANDOM_STATE,
            max_iter=1000, n_jobs=-1
        ),

        'RandomForest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),

        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=5
        ),

        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128), random_state=RANDOM_STATE,
            max_iter=500, early_stopping=True
        ),

        'Bagging_SVM': BaggingClassifier(
            estimator=LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000),
            n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for name, model in models.items():
        result = train_and_evaluate(name, model, X_train, X_test, y_train, y_test, le)
        print(f"\n{name}:")
        print(f"  准确率: {result['accuracy']:.4f}, F1_macro: {result['f1_macro']:.4f}")
        print(f"  召回率: {result['recalls']}")
        all_results.append(result)

    # ============ 高级优化方法 ============
    print("\n" + "=" * 70)
    print("3. 高级优化方法")
    print("=" * 70)

    # Focal Loss
    print("\n--- Focal Loss Logistic Regression ---")
    focal_model = FocalLossLogisticRegression(gamma=2.0, n_iterations=3)
    result = train_and_evaluate("FocalLoss_LR", focal_model, X_train, X_test, y_train, y_test, le)
    print(f"  准确率: {result['accuracy']:.4f}, F1_macro: {result['f1_macro']:.4f}")
    print(f"  召回率: {result['recalls']}")
    all_results.append(result)

    # 自适应阈值
    print("\n--- 自适应阈值分类器 ---")
    adaptive_model = AdaptiveThresholdClassifier(
        LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
    )
    result = train_and_evaluate("AdaptiveThreshold", adaptive_model, X_train, X_test, y_train, y_test, le)
    print(f"  准确率: {result['accuracy']:.4f}, F1_macro: {result['f1_macro']:.4f}")
    print(f"  召回率: {result['recalls']}")
    all_results.append(result)

    # 投票集成
    print("\n--- 投票集成 ---")
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=3000)),
            ('lr', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000, C=10)),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)),
        ],
        voting='hard'
    )
    result = train_and_evaluate("VotingEnsemble", voting_clf, X_train, X_test, y_train, y_test, le)
    print(f"  准确率: {result['accuracy']:.4f}, F1_macro: {result['f1_macro']:.4f}")
    print(f"  召回率: {result['recalls']}")
    all_results.append(result)

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("结果汇总 (按 Macro F1 排序)")
    print("=" * 70)

    results_df = pd.DataFrame([
        {
            '模型': r['name'],
            '准确率': r['accuracy'],
            'F1_weighted': r['f1_weighted'],
            'F1_macro': r['f1_macro'],
            '协调能力召回': r['recalls'].get('协调能力', r['recalls'].get(0, 0)),
            '合规能力召回': r['recalls'].get('合规能力', r['recalls'].get(1, 0)),
            '提取能力召回': r['recalls'].get('提取能力', r['recalls'].get(2, 0)),
        }
        for r in all_results
    ])

    results_df = results_df.sort_values('F1_macro', ascending=False)

    print(f"\n{'排名':<4} {'模型':<30} {'准确率':<8} {'F1_macro':<10} {'合规召回':<10} {'提取召回':<10}")
    print("-" * 82)

    for i, (idx, row) in enumerate(results_df.head(15).iterrows()):
        print(f"{i+1:<4} {row['模型']:<30} {row['准确率']:.4f}   {row['F1_macro']:.4f}     "
              f"{row['合规能力召回']:.4f}     {row['提取能力召回']:.4f}")

    # 最佳模型
    best = results_df.iloc[0]
    baseline = results_df[results_df['模型'] == 'SVM + tfidf_word_1_3'].iloc[0] if 'SVM + tfidf_word_1_3' in results_df['模型'].values else results_df.iloc[-1]

    print(f"\n" + "=" * 70)
    print(f"最佳纯ML模型: {best['模型']}")
    print(f"  准确率: {best['准确率']:.4f}")
    print(f"  F1 macro: {best['F1_macro']:.4f}")
    print(f"  合规能力召回率: {best['合规能力召回']:.4f}")
    print(f"  提取能力召回率: {best['提取能力召回']:.4f}")
    print("=" * 70)

    # 保存结果
    results_df.to_csv('pure_ml_comparison.csv', index=False)
    print(f"\n结果已保存到 pure_ml_comparison.csv")

    return results_df


if __name__ == '__main__':
    main()
