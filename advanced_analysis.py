#!/usr/bin/env python3
"""
高级分析脚本：
1. 详细错误分析
2. 类别不平衡处理
3. 对2014年数据进行分类预测
"""

import pandas as pd
import numpy as np
import glob
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 尝试导入 imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imblearn 未安装，将使用类别权重处理不平衡")

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_training_data():
    """加载训练数据"""
    print("=" * 70)
    print("1. 加载训练数据 (2015年分类结果)")
    print("=" * 70)

    all_files = glob.glob('classification_batch*.csv')
    print(f"找到 {len(all_files)} 个批次文件")

    dfs = []
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            pass

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['序号'])

    # 过滤掉"其他"类别（样本太少）
    df = df[df['分类'] != '其他']

    print(f"总记录数: {len(df)}")
    print(f"\n类别分布:")
    for cls, count in df['分类'].value_counts().items():
        print(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

    return df


def prepare_features(df):
    """准备特征"""
    df['text'] = df['采购人'].fillna('') + ' ' + df['合同名称'].fillna('')
    return df


def detailed_error_analysis(model, vectorizer, X_test, y_test, le, df_test):
    """详细错误分析"""
    print("\n" + "=" * 70)
    print("2. 详细错误分析")
    print("=" * 70)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # 基本统计
    correct = (y_test == y_pred).sum()
    total = len(y_test)
    print(f"\n正确预测: {correct} / {total} ({correct/total*100:.2f}%)")
    print(f"错误预测: {total-correct} / {total} ({(total-correct)/total*100:.2f}%)")

    # 混淆矩阵分析
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    classes = le.classes_

    header = '实际\\预测'
    print(f"\n{header:12}", end='')
    for c in classes:
        print(f"{c:>12}", end='')
    print()
    print("-" * (12 + 12 * len(classes)))

    for i, cls in enumerate(classes):
        print(f"{cls:12}", end='')
        for j in range(len(classes)):
            print(f"{cm[i,j]:>12}", end='')
        print()

    # 各类别详细分析
    print("\n各类别详细指标:")
    print(f"{'类别':12} {'精确率':>10} {'召回率':>10} {'F1':>10} {'支持数':>10}")
    print("-" * 52)

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    for cls in classes:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f1 = report[cls]['f1-score']
        s = report[cls]['support']
        print(f"{cls:12} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {int(s):>10}")

    # 错误类型分析
    print("\n错误类型详细分析:")
    mask = y_test != y_pred
    errors = df_test[mask].copy()
    errors['实际'] = le.inverse_transform(y_test[mask])
    errors['预测'] = le.inverse_transform(y_pred[mask])

    error_pairs = errors.groupby(['实际', '预测']).size().reset_index(name='count')
    error_pairs = error_pairs.sort_values('count', ascending=False)

    print(f"\n{'实际类别':12} {'被误判为':12} {'数量':>8} {'占错误比例':>12}")
    print("-" * 44)
    total_errors = len(errors)
    for _, row in error_pairs.iterrows():
        pct = row['count'] / total_errors * 100
        print(f"{row['实际']:12} {row['预测']:12} {row['count']:>8} {pct:>11.2f}%")

    # 典型错误案例
    print("\n典型错误案例分析:")
    for (actual, pred), group in errors.groupby(['实际', '预测']):
        print(f"\n【{actual} → {pred}】共 {len(group)} 例:")
        for idx, row in group.head(3).iterrows():
            buyer = str(row['采购人'])[:20]
            contract = str(row['合同名称'])[:35]
            print(f"  • {buyer} | {contract}...")

    return errors


def handle_imbalance(X_train, y_train, method='class_weight'):
    """处理类别不平衡"""
    print("\n" + "=" * 70)
    print("3. 类别不平衡处理")
    print("=" * 70)

    print(f"\n原始训练集类别分布:")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  类别 {cls}: {count}")

    results = {}

    # 方法1: 类别权重
    print("\n方法1: 使用类别权重 (class_weight='balanced')")
    model_weighted = LinearSVC(max_iter=3000, random_state=RANDOM_STATE, class_weight='balanced')
    results['class_weight'] = model_weighted

    # 方法2: 过采样 (如果有imblearn)
    if HAS_IMBLEARN:
        print("\n方法2: SMOTE过采样")
        # 注意：SMOTE需要密集矩阵，对于大型稀疏矩阵可能内存不足
        # 这里使用RandomOverSampler作为替代
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        print(f"过采样后类别分布:")
        for cls, count in sorted(Counter(y_resampled).items()):
            print(f"  类别 {cls}: {count}")
        results['oversampling'] = (X_resampled, y_resampled)

    return results


def train_with_imbalance_handling(X_train, X_test, y_train, y_test, le, imbalance_results):
    """使用不同方法训练并比较"""
    print("\n比较不同不平衡处理方法的效果:")
    print(f"\n{'方法':20} {'准确率':>10} {'F1 (weighted)':>15} {'F1 (macro)':>12}")
    print("-" * 60)

    best_model = None
    best_f1 = 0
    best_method = None

    # 基线：无处理
    model_base = LinearSVC(max_iter=3000, random_state=RANDOM_STATE)
    model_base.fit(X_train, y_train)
    y_pred = model_base.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')
    print(f"{'无处理 (基线)':20} {acc:>10.4f} {f1_w:>15.4f} {f1_m:>12.4f}")

    # 类别权重
    model_weighted = imbalance_results['class_weight']
    model_weighted.fit(X_train, y_train)
    y_pred = model_weighted.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')
    print(f"{'类别权重':20} {acc:>10.4f} {f1_w:>15.4f} {f1_m:>12.4f}")

    if f1_m > best_f1:
        best_f1 = f1_m
        best_model = model_weighted
        best_method = '类别权重'

    # 过采样
    if 'oversampling' in imbalance_results:
        X_res, y_res = imbalance_results['oversampling']
        model_os = LinearSVC(max_iter=3000, random_state=RANDOM_STATE)
        model_os.fit(X_res, y_res)
        y_pred = model_os.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')
        print(f"{'随机过采样':20} {acc:>10.4f} {f1_w:>15.4f} {f1_m:>12.4f}")

        if f1_m > best_f1:
            best_f1 = f1_m
            best_model = model_os
            best_method = '随机过采样'

    print(f"\n最佳方法: {best_method} (Macro F1: {best_f1:.4f})")

    return best_model, best_method


def load_2014_data():
    """加载2014年数据"""
    print("\n" + "=" * 70)
    print("4. 加载2014年数据")
    print("=" * 70)

    try:
        # 尝试读取Stata文件
        df = pd.read_stata('2014.dta')
        print(f"成功读取 2014.dta")
        print(f"记录数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"读取2014.dta失败: {e}")
        return None


def predict_2014_data(model, vectorizer, le, df_2014):
    """对2014年数据进行分类预测"""
    print("\n" + "=" * 70)
    print("5. 对2014年数据进行分类预测")
    print("=" * 70)

    # 检查列名并构建文本特征
    print(f"\n2014年数据列名: {list(df_2014.columns)}")

    # 直接使用正确的列名
    buyer_col = '采购人'
    contract_col = '合同名称'

    if buyer_col not in df_2014.columns or contract_col not in df_2014.columns:
        print(f"列名不存在，尝试查找...")
        for col in df_2014.columns:
            if col == '采购人':
                buyer_col = col
            if col == '合同名称':
                contract_col = col

    print(f"\n使用列: 采购人='{buyer_col}', 合同名称='{contract_col}'")

    # 构建文本特征
    df_2014['text'] = df_2014[buyer_col].fillna('').astype(str) + ' ' + df_2014[contract_col].fillna('').astype(str)

    # 向量化
    X_2014 = vectorizer.transform(df_2014['text'])

    # 预测
    y_pred = model.predict(X_2014)
    df_2014['预测分类'] = le.inverse_transform(y_pred)

    # 预测结果统计
    print("\n预测结果分布:")
    pred_counts = df_2014['预测分类'].value_counts()
    for cls, count in pred_counts.items():
        pct = count / len(df_2014) * 100
        print(f"  {cls}: {count} ({pct:.2f}%)")

    # 显示部分预测结果
    print("\n预测结果样例 (每类5条):")
    for cls in le.classes_:
        samples = df_2014[df_2014['预测分类'] == cls].head(5)
        print(f"\n【{cls}】:")
        for idx, row in samples.iterrows():
            buyer = str(row[buyer_col])[:25] if pd.notna(row[buyer_col]) else ''
            contract = str(row[contract_col])[:35] if pd.notna(row[contract_col]) else ''
            print(f"  • {buyer} | {contract}")

    return df_2014


def save_results(df_2014, output_file='2014_classified_by_ml.csv'):
    """保存预测结果"""
    print(f"\n保存预测结果到: {output_file}")
    df_2014.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(df_2014)} 条记录")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("政府采购合同分类 - 高级分析与2014年数据预测")
    print("=" * 70)

    # 1. 加载训练数据
    df = load_training_data()
    df = prepare_features(df)

    # 准备标签
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['分类'])

    # 划分数据
    X = df['text']
    y = df['label'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 向量化
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=15000,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"\n训练集: {X_train_vec.shape[0]}, 测试集: {X_test_vec.shape[0]}")
    print(f"特征维度: {X_train_vec.shape[1]}")

    # 训练基础模型
    print("\n训练基础 Linear SVM 模型...")
    model = LinearSVC(max_iter=3000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    # 2. 详细错误分析
    df_test = df.loc[idx_test].copy()
    errors = detailed_error_analysis(model, vectorizer, X_test, y_test, le, df_test)

    # 3. 处理类别不平衡
    imbalance_results = handle_imbalance(X_train_vec, y_train)
    best_model, best_method = train_with_imbalance_handling(
        X_train_vec, X_test_vec, y_train, y_test, le, imbalance_results
    )

    # 4. 加载并预测2014年数据
    df_2014 = load_2014_data()

    if df_2014 is not None:
        df_2014_predicted = predict_2014_data(best_model, vectorizer, le, df_2014)

        # 5. 保存结果
        save_results(df_2014_predicted)

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
