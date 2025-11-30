#!/usr/bin/env python3
"""
政府采购合同国家能力分类 - 机器学习训练脚本
分类类别：提取能力、协调能力、合规能力
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
import re
from collections import Counter

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 尝试导入 XGBoost 和 LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost 未安装，将跳过该模型")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM 未安装，将跳过该模型")

warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_all_data():
    """加载并合并所有分类批次数据"""
    print("=" * 60)
    print("1. 加载数据")
    print("=" * 60)

    all_files = glob.glob('classification_batch*.csv')
    print(f"找到 {len(all_files)} 个批次文件")

    dfs = []
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"总记录数: {len(combined_df)}")

    # 去除重复项
    combined_df = combined_df.drop_duplicates(subset=['序号'])
    print(f"去重后记录数: {len(combined_df)}")

    return combined_df


def analyze_data(df):
    """数据分析"""
    print("\n" + "=" * 60)
    print("2. 数据分析")
    print("=" * 60)

    # 类别分布
    print("\n类别分布:")
    class_counts = df['分类'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count} ({pct:.2f}%)")

    # 文本长度统计
    df['text'] = df['采购人'].fillna('') + ' ' + df['合同名称'].fillna('')
    df['text_len'] = df['text'].str.len()

    print(f"\n文本长度统计:")
    print(f"  平均长度: {df['text_len'].mean():.2f}")
    print(f"  最短: {df['text_len'].min()}")
    print(f"  最长: {df['text_len'].max()}")

    return df


def simple_tokenize(text):
    """简单的中文分词：使用字符级别 + 常见词组"""
    if pd.isna(text):
        return ""

    # 提取中文字符和数字
    text = str(text)

    # 常见机构关键词（作为整体保留）
    keywords = [
        '公安局', '检察院', '法院', '监狱', '税务局', '海关', '统计局',
        '卫生局', '教育局', '财政局', '医院', '学校', '大学', '中学', '小学',
        '幼儿园', '公路局', '交通局', '环保局', '气象局', '水利局',
        '农业局', '林业局', '国土局', '规划局', '住建局', '民政局',
        '人社局', '市场监管', '食药监', '质监局', '工商局', '消防',
        '武警', '部队', '军区', '司法局', '城管局', '安监局',
        '发改委', '经信委', '科技局', '文化局', '体育局', '旅游局',
        '人民银行', '银监局', '证监局', '保监局', '外汇管理',
        '地质调查', '测绘局', '档案局', '保密局', '信访局',
        '纪委', '监察委', '审计局', '统战部', '组织部', '宣传部'
    ]

    # 先替换关键词为特殊标记
    for i, kw in enumerate(keywords):
        text = text.replace(kw, f' KW{i} ')

    # 对剩余文本进行字符级分割
    chars = list(text)

    # 还原关键词
    result = ' '.join(chars)
    for i, kw in enumerate(keywords):
        result = result.replace(f'K W {i}', kw)
        result = result.replace(f'KW{i}', kw)

    return result


def prepare_features(df):
    """特征工程"""
    print("\n" + "=" * 60)
    print("3. 特征工程")
    print("=" * 60)

    # 合并文本
    df['text'] = df['采购人'].fillna('') + ' ' + df['合同名称'].fillna('')

    # 简单分词处理
    print("正在进行文本处理...")
    df['text_processed'] = df['text'].apply(simple_tokenize)

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['分类'])

    print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    return df, le


def create_vectorizers(X_train, X_test):
    """创建多种文本向量化器"""
    vectorizers = {}

    # TF-IDF (字符级 n-gram)
    print("创建 TF-IDF (字符级) 向量化器...")
    tfidf_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 4),
        max_features=15000,
        min_df=2
    )
    X_train_tfidf_char = tfidf_char.fit_transform(X_train)
    X_test_tfidf_char = tfidf_char.transform(X_test)
    vectorizers['tfidf_char'] = (X_train_tfidf_char, X_test_tfidf_char, tfidf_char)

    # TF-IDF (词级，空格分隔)
    print("创建 TF-IDF (词级) 向量化器...")
    tfidf_word = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=15000,
        min_df=2,
        token_pattern=r'(?u)\b\w+\b'
    )
    X_train_tfidf_word = tfidf_word.fit_transform(X_train)
    X_test_tfidf_word = tfidf_word.transform(X_test)
    vectorizers['tfidf_word'] = (X_train_tfidf_word, X_test_tfidf_word, tfidf_word)

    # 词袋模型 (字符级)
    print("创建词袋模型向量化器...")
    count_vec = CountVectorizer(
        analyzer='char',
        ngram_range=(1, 3),
        max_features=15000,
        min_df=2
    )
    X_train_count = count_vec.fit_transform(X_train)
    X_test_count = count_vec.transform(X_test)
    vectorizers['count_char'] = (X_train_count, X_test_count, count_vec)

    print(f"\nTF-IDF (字符级) 特征维度: {X_train_tfidf_char.shape[1]}")
    print(f"TF-IDF (词级) 特征维度: {X_train_tfidf_word.shape[1]}")
    print(f"词袋模型 特征维度: {X_train_count.shape[1]}")

    return vectorizers


def get_models():
    """获取所有待训练的模型"""
    models = {
        # 线性模型
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1, C=1.0
        ),
        'Logistic Regression (L1)': LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Logistic Regression (C=10)': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1, C=10.0
        ),

        # SVM
        'Linear SVM': LinearSVC(
            max_iter=3000, random_state=RANDOM_STATE, C=1.0
        ),
        'Linear SVM (C=0.1)': LinearSVC(
            max_iter=3000, random_state=RANDOM_STATE, C=0.1
        ),

        # 朴素贝叶斯
        'Multinomial NB': MultinomialNB(alpha=0.1),
        'Multinomial NB (alpha=1)': MultinomialNB(alpha=1.0),
        'Complement NB': ComplementNB(alpha=0.1),

        # 树模型
        'Decision Tree': DecisionTreeClassifier(
            max_depth=30, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Random Forest (200)': RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),

        # KNN
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),

        # 神经网络
        'MLP (256-128)': MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=500,
            random_state=RANDOM_STATE, early_stopping=True
        ),
        'MLP (512-256)': MLPClassifier(
            hidden_layer_sizes=(512, 256), max_iter=500,
            random_state=RANDOM_STATE, early_stopping=True
        ),
    }

    # 添加 XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False,
            eval_metric='mlogloss'
        )
        models['XGBoost (200)'] = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False,
            eval_metric='mlogloss'
        )

    # 添加 LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        models['LightGBM (200)'] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )

    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test, vectorizer_name, le):
    """训练并评估所有模型"""
    results = []

    for name, model in models.items():
        try:
            # 训练
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 计算指标
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'model': name,
                'vectorizer': vectorizer_name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

            print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")

        except Exception as e:
            print(f"  {name}: 训练失败 - {e}")

    return results


def cross_validate_best_models(best_configs, X, y, cv=5):
    """对最佳模型进行交叉验证"""
    print("\n" + "=" * 60)
    print("5. 交叉验证 (最佳模型)")
    print("=" * 60)

    cv_results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    for config in best_configs[:5]:  # 取前5个最佳模型
        model_name = config['model']
        vec_name = config['vectorizer']

        # 重新创建模型和向量化器
        models = get_models()
        if model_name not in models:
            continue

        model = models[model_name]

        # 创建向量化器
        if vec_name == 'tfidf_char':
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=15000, min_df=2)
        elif vec_name == 'tfidf_word':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=15000, min_df=2)
        else:
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=15000, min_df=2)

        X_vec = vectorizer.fit_transform(X)

        # 交叉验证
        scores = cross_val_score(model, X_vec, y, cv=skf, scoring='f1_weighted', n_jobs=-1)

        cv_results.append({
            'model': model_name,
            'vectorizer': vec_name,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores
        })

        print(f"{model_name} + {vec_name}:")
        print(f"  CV F1: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        print(f"  各折: {[f'{s:.4f}' for s in scores]}")

    return cv_results


def detailed_analysis(best_config, X_train, X_test, y_train, y_test, le):
    """对最佳模型进行详细分析"""
    print("\n" + "=" * 60)
    print("6. 最佳模型详细分析")
    print("=" * 60)

    model_name = best_config['model']
    vec_name = best_config['vectorizer']

    print(f"\n最佳模型: {model_name}")
    print(f"向量化方法: {vec_name}")

    # 重新训练最佳模型
    models = get_models()
    model = models[model_name]

    if vec_name == 'tfidf_char':
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=15000, min_df=2)
    elif vec_name == 'tfidf_word':
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=15000, min_df=2)
    else:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=15000, min_df=2)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':15} " + " ".join([f"{c:>10}" for c in le.classes_]))
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:15} " + " ".join([f"{v:>10}" for v in row]))

    # 各类别正确率
    print("\n各类别正确率:")
    for i, cls in enumerate(le.classes_):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"  {cls}: {acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

    return model, vectorizer


def analyze_errors(model, vectorizer, X_test, y_test, le, df_test):
    """错误分析"""
    print("\n" + "=" * 60)
    print("7. 错误分析")
    print("=" * 60)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # 找出错误预测
    mask = y_test != y_pred
    errors = df_test[mask].copy()
    errors['预测'] = le.inverse_transform(y_pred[mask])
    errors['实际'] = le.inverse_transform(y_test[mask])

    print(f"\n总错误数: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.2f}%)")

    # 错误类型分布
    print("\n错误类型分布:")
    error_types = errors.groupby(['实际', '预测']).size().reset_index(name='count')
    error_types = error_types.sort_values('count', ascending=False)
    for _, row in error_types.iterrows():
        print(f"  {row['实际']} -> {row['预测']}: {row['count']}例")

    # 显示一些错误样例
    print("\n错误样例 (前10条):")
    for idx, row in errors.head(10).iterrows():
        contract_name = str(row['合同名称'])[:35] if pd.notna(row['合同名称']) else ''
        print(f"  [{row['序号']}] {row['采购人']} - {contract_name}...")
        print(f"       实际: {row['实际']} | 预测: {row['预测']}")

    return errors


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("政府采购合同 - 国家能力分类模型训练")
    print("=" * 60)

    # 1. 加载数据
    df = load_all_data()

    # 2. 数据分析
    df = analyze_data(df)

    # 3. 特征工程
    df, le = prepare_features(df)

    # 4. 划分训练集和测试集
    print("\n" + "=" * 60)
    print("4. 模型训练与评估")
    print("=" * 60)

    X = df['text_processed']
    y = df['label'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 创建向量化器
    vectorizers = create_vectorizers(X_train, X_test)

    # 获取模型
    models = get_models()
    print(f"\n待训练模型数: {len(models)}")

    # 训练和评估
    all_results = []

    for vec_name, (X_tr, X_te, vec) in vectorizers.items():
        print(f"\n使用 {vec_name} 向量化:")
        results = train_and_evaluate(models, X_tr, X_te, y_train, y_test, vec_name, le)
        all_results.extend(results)

    # 汇总结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('f1', ascending=False)

    print("\n" + "=" * 60)
    print("模型性能排名 (按F1分数)")
    print("=" * 60)
    print(f"\n{'排名':<4} {'模型':<28} {'向量化':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 100)

    for i, row in results_df.head(20).iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"{rank:<4} {row['model']:<28} {row['vectorizer']:<12} {row['accuracy']:.4f}     {row['precision']:.4f}     {row['recall']:.4f}     {row['f1']:.4f}")

    # 5. 交叉验证最佳模型
    best_configs = results_df.head(5).to_dict('records')
    cv_results = cross_validate_best_models(best_configs, X, y)

    # 6. 详细分析最佳模型
    best_config = results_df.iloc[0].to_dict()

    # 获取正确的向量化数据
    vec_name = best_config['vectorizer']
    X_tr, X_te, vec = vectorizers[vec_name]

    model, vectorizer = detailed_analysis(best_config, X_train, X_test, y_train, y_test, le)

    # 7. 错误分析
    df_test = df.loc[idx_test].copy()
    errors = analyze_errors(model, vectorizer, X_test, y_test, le, df_test)

    # 保存结果
    results_df.to_csv('model_comparison_results.csv', index=False)
    errors.to_csv('error_analysis.csv', index=False)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print("\n结果文件:")
    print("  - model_comparison_results.csv: 所有模型对比结果")
    print("  - error_analysis.csv: 错误分析详情")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    best = results_df.iloc[0]
    print(f"\n最佳模型: {best['model']} + {best['vectorizer']}")
    print(f"测试集性能:")
    print(f"  - 准确率: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"  - 精确率: {best['precision']:.4f}")
    print(f"  - 召回率: {best['recall']:.4f}")
    print(f"  - F1分数: {best['f1']:.4f}")

    if cv_results:
        cv_best = cv_results[0]
        print(f"\n交叉验证性能 ({cv_best['model']}):")
        print(f"  - CV F1: {cv_best['cv_mean']:.4f} (+/- {cv_best['cv_std']*2:.4f})")


if __name__ == '__main__':
    main()
