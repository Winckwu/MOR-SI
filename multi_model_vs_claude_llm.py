#!/usr/bin/env python3
"""
多模型与Claude LLM理解分类对比
验证各模型与Claude LLM分类的一致率
"""

import pandas as pd
import numpy as np
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 加载训练数据 ==========
print("=" * 70)
print("加载训练数据...")
print("=" * 70)

all_files = glob.glob('classification_batch*.csv')
dfs = []
for file in sorted(all_files):
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        dfs.append(df)
    except Exception as e:
        print(f"  警告: {e}")

train_df = pd.concat(dfs, ignore_index=True)
train_df = train_df.drop_duplicates(subset=['序号'])
train_df = train_df[train_df['分类'] != '其他']
train_df['text'] = train_df['采购人'].fillna('') + ' ' + train_df['合同名称'].fillna('')

print(f"训练数据: {len(train_df)} 条")

# ========== 2. 加载Claude LLM分类结果 ==========
print("\n" + "=" * 70)
print("加载Claude LLM理解分类结果...")
print("=" * 70)

# 从已保存的对比文件加载Claude分类
claude_df = pd.read_csv('claude_llm_vs_ml_comparison.csv')
print(f"2012-2014年数据: {len(claude_df)} 条")
print(f"Claude LLM分类分布:")
print(claude_df['Claude_LLM分类'].value_counts())

# ========== 3. 准备测试数据 ==========
# 构建测试文本
claude_df['text'] = claude_df['采购人'].fillna('').astype(str) + ' ' + claude_df['合同名称'].fillna('').astype(str)

# ========== 4. 向量化 ==========
print("\n" + "=" * 70)
print("TF-IDF向量化 (char 2-4 grams)...")
print("=" * 70)

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=15000,
    min_df=2
)

X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(claude_df['text'])

le = LabelEncoder()
y_train = le.fit_transform(train_df['分类'])
y_claude = claude_df['Claude_LLM分类'].values

print(f"训练特征: {X_train.shape}")
print(f"测试特征: {X_test.shape}")

# ========== 5. 训练多个模型并对比 ==========
print("\n" + "=" * 70)
print("训练多个模型并与Claude LLM分类对比...")
print("=" * 70)

models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'LinearSVC': LinearSVC(class_weight='balanced', random_state=42, max_iter=3000),
    'LinearSVC_C0.1': LinearSVC(class_weight='balanced', C=0.1, random_state=42, max_iter=3000),
    'LinearSVC_C10': LinearSVC(class_weight='balanced', C=10, random_state=42, max_iter=3000),
    'LogisticRegression_C10': LogisticRegression(class_weight='balanced', C=10, random_state=42, max_iter=1000),
    'SGD_hinge': SGDClassifier(loss='hinge', class_weight='balanced', random_state=42, max_iter=1000),
    'SGD_log': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'RandomForest_200': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'MultinomialNB': MultinomialNB(alpha=0.1),
}

results = []

for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # 计算一致率
    agree = (y_pred_labels == y_claude).sum()
    total = len(y_claude)
    rate = agree / total * 100

    results.append({
        '模型': name,
        '一致数': agree,
        '总数': total,
        '与Claude LLM一致率': rate
    })

    print(f"  一致: {agree}/{total} ({rate:.2f}%)")

    # 保存每个模型的预测结果
    claude_df[f'{name}_预测'] = y_pred_labels

# ========== 6. 结果排序并保存 ==========
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('与Claude LLM一致率', ascending=False)

print("\n" + "=" * 70)
print("多模型与Claude LLM理解分类一致率排名")
print("=" * 70)

for i, (_, row) in enumerate(results_df.iterrows(), 1):
    if i == 1:
        rank = "🥇"
    elif i == 2:
        rank = "🥈"
    elif i == 3:
        rank = "🥉"
    else:
        rank = f"{i}."
    print(f"{rank} {row['模型']}: {row['一致数']}/{row['总数']} ({row['与Claude LLM一致率']:.2f}%)")

# 保存结果
results_df.to_csv('multi_model_vs_claude_llm_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: multi_model_vs_claude_llm_comparison.csv")

# 保存详细对比
claude_df.to_csv('all_models_predictions_vs_claude_llm.csv', index=False, encoding='utf-8-sig')
print(f"详细预测已保存到: all_models_predictions_vs_claude_llm.csv")

# ========== 7. 分析不一致记录 ==========
print("\n" + "=" * 70)
print("最佳模型 vs Claude LLM 不一致分析")
print("=" * 70)

best_model = results_df.iloc[0]['模型']
disagree = claude_df[claude_df[f'{best_model}_预测'] != claude_df['Claude_LLM分类']]

print(f"\n{best_model} 与 Claude LLM 不一致的记录 ({len(disagree)}条):")
for idx, row in disagree.iterrows():
    buyer = str(row['采购人'])[:30]
    print(f"  {row.get('年份', '')}: {buyer}")
    print(f"    ML:{row[f'{best_model}_预测']} vs Claude:{row['Claude_LLM分类']}")
