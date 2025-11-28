#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府采购合同国家能力分类器 V2
State Capacity Contract Classifier V2

改进版本，包含：
- 更精细的关键词匹配
- 更强的特征工程
- 深度学习模型（简单神经网络）
- 数据增强和类别平衡处理

基于 Berwick & Christia (2018) 《State Capacity Redux》论文框架
"""

import pandas as pd
import numpy as np
import re
import warnings
import json
from collections import Counter
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 第一部分：改进的分类规则定义
# ============================================================================

class ImprovedStateCapacityLabeler:
    """
    改进版标注器，使用更精细的规则和加权机制
    """

    def __init__(self):
        # 汲取能力关键词（权重加强）
        self.extractive_patterns = {
            # 强信号（权重3）
            'strong': [
                '税务', '税收', '财税', '地税', '国税', '财政局', '财政厅',
                '审计', '审计局', '会计', '出纳', '预算', '决算',
                '资产评估', '资产清查', '国有资产', '资产管理',
                '土地储备', '土地出让', '矿产资源', '矿权',
                '征收', '征地', '拆迁补偿',
            ],
            # 中等信号（权重2）
            'medium': [
                '财务', '资金', '收费', '罚款', '罚没',
                '资产', '产权', '不动产', '房产登记',
                '国土', '土地', '测绘', '地籍',
                '银行', '金融', '贷款',
            ],
            # 弱信号（权重1）
            'weak': [
                '评估', '鉴定价格', '价值评定',
            ]
        }

        # 协调能力关键词
        self.coordination_patterns = {
            'strong': [
                # 基础设施建设
                '道路建设', '公路建设', '桥梁', '隧道', '市政工程',
                '水利工程', '电网', '供水', '排水', '管网',
                '通信基站', '网络建设', '信息化建设', '电子政务',
                # 政府办公
                '政府采购', '办公设备', '办公家具', '公务用车',
            ],
            'medium': [
                '道路', '公路', '交通', '运输', '电力', '供电',
                '通讯', '通信', '网络', '信息化', '数字化',
                '建设', '工程', '施工', '改造', '修缮', '维修',
                '装修', '装饰', '绿化', '环卫',
                '办公', '会议', '档案', '印刷', '车辆', '后勤',
                '规划', '设计', '咨询', '监理',
            ],
            'weak': [
                '设备', '家具', '打印', '复印', '空调', '电脑', '计算机',
            ]
        }

        # 合规能力关键词
        self.compliance_patterns = {
            'strong': [
                # 教育服务
                '学校', '教育局', '教委', '教学设备', '教学仪器',
                '多媒体教学', '录播系统', '实验室', '实训室',
                '图书馆', '图书采购', '教材', '课本',
                # 医疗卫生
                '医院', '卫生院', '疾控中心', '医疗设备', '医疗器械',
                'CT', 'MRI', 'B超', '彩超', '手术', '诊断',
                '药品', '医药', '疫苗', '防疫',
                # 公共安全
                '公安局', '派出所', '警务', '执法', '司法局',
                '监控系统', '安防系统', '消防', '应急',
            ],
            'medium': [
                '教育', '教学', '培训', '学习', '课程',
                '医疗', '卫生', '健康', '诊疗', '护理', '康复',
                '检验', '检测', '化验', '影像',
                '安防', '监控', '安保', '保安',
                '环保', '环境', '污染', '垃圾处理',
                '养老', '福利', '救助', '社区服务',
            ],
            'weak': [
                '体育', '运动', '健身', '文化', '艺术',
            ]
        }

        # 行业强映射
        self.industry_strong_mapping = {
            # 汲取能力
            '财政': 'extractive',
            '税务': 'extractive',
            '金融业': 'extractive',

            # 合规能力（教育医疗）
            '普通高等教育': 'compliance',
            '中等职业学校教育': 'compliance',
            '普通小学教育': 'compliance',
            '普通初中教育': 'compliance',
            '学前教育': 'compliance',
            '特殊教育': 'compliance',
            '综合医院': 'compliance',
            '专科医院': 'compliance',
            '疾病预防控制中心': 'compliance',
            '卫生和社会工作': 'compliance',
        }

    def _calculate_score(self, text, patterns):
        """计算加权得分"""
        if pd.isna(text):
            return 0
        text = str(text).lower()

        score = 0
        matched_keywords = []

        for keyword in patterns.get('strong', []):
            if keyword.lower() in text:
                score += 3
                matched_keywords.append((keyword, 3))

        for keyword in patterns.get('medium', []):
            if keyword.lower() in text:
                score += 2
                matched_keywords.append((keyword, 2))

        for keyword in patterns.get('weak', []):
            if keyword.lower() in text:
                score += 1
                matched_keywords.append((keyword, 1))

        return score, matched_keywords

    def label_single(self, contract_name, subject_name=None, industry=None, purchaser=None):
        """对单条记录进行标注"""

        # 合并所有可用文本
        texts = []
        if contract_name and not pd.isna(contract_name):
            texts.append(str(contract_name))
        if subject_name and not pd.isna(subject_name):
            texts.append(str(subject_name))
        if industry and not pd.isna(industry):
            texts.append(str(industry))
        if purchaser and not pd.isna(purchaser):
            texts.append(str(purchaser))

        full_text = ' '.join(texts)

        # 计算各类别得分
        ext_score, ext_keywords = self._calculate_score(full_text, self.extractive_patterns)
        coord_score, coord_keywords = self._calculate_score(full_text, self.coordination_patterns)
        comp_score, comp_keywords = self._calculate_score(full_text, self.compliance_patterns)

        # 行业强映射加成
        industry_label = None
        if industry and not pd.isna(industry):
            for ind_key, label in self.industry_strong_mapping.items():
                if ind_key in str(industry):
                    industry_label = label
                    if label == 'extractive':
                        ext_score += 5
                    elif label == 'coordination':
                        coord_score += 5
                    elif label == 'compliance':
                        comp_score += 5
                    break

        # 采购人类型判断（补充规则）
        if purchaser and not pd.isna(purchaser):
            purchaser_str = str(purchaser).lower()
            if any(kw in purchaser_str for kw in ['学校', '大学', '学院', '小学', '中学', '幼儿园', '教育']):
                comp_score += 3
            elif any(kw in purchaser_str for kw in ['医院', '卫生院', '疾控', '卫生']):
                comp_score += 3
            elif any(kw in purchaser_str for kw in ['税务', '财政', '审计', '国土']):
                ext_score += 3
            elif any(kw in purchaser_str for kw in ['住建', '交通', '公路', '市政']):
                coord_score += 2

        scores = {
            'extractive': ext_score,
            'coordination': coord_score,
            'compliance': comp_score
        }

        max_score = max(scores.values())
        total_score = sum(scores.values())

        if max_score == 0:
            # 无明显特征，默认为协调能力（通用政府采购）
            return 'coordination', 0.1, '默认分类'

        label = max(scores, key=scores.get)

        # 置信度计算（考虑区分度）
        second_score = sorted(scores.values(), reverse=True)[1]
        if total_score > 0:
            confidence = (max_score - second_score + 1) / (total_score + 1)
        else:
            confidence = 0.1

        # 生成原因说明
        reason = f"E:{ext_score},C:{coord_score},P:{comp_score}"

        return label, min(confidence, 1.0), reason

    def label_dataframe(self, df):
        """对整个数据框进行标注"""
        labels = []
        confidences = []
        reasons = []

        for idx, row in df.iterrows():
            label, conf, reason = self.label_single(
                row.get('合同名称', ''),
                row.get('主要标的名称', ''),
                row.get('所属行业', ''),
                row.get('采购人', '')
            )
            labels.append(label)
            confidences.append(conf)
            reasons.append(reason)

        df_labeled = df.copy()
        df_labeled['capacity_label'] = labels
        df_labeled['label_confidence'] = confidences
        df_labeled['label_reason'] = reasons

        return df_labeled


# ============================================================================
# 第二部分：数据处理
# ============================================================================

def load_data():
    """加载数据"""
    print("=" * 60)
    print("📂 加载数据...")
    print("=" * 60)

    dfs = []
    for year in ['2012', '2013', '2014']:
        df = pd.read_stata(f'{year}.dta')
        print(f"  {year}年: {len(df)} 条记录")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n  总计: {len(df_all)} 条记录")
    return df_all


def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_combined_text(row):
    """创建组合文本"""
    texts = []
    for col in ['合同名称', '主要标的名称', '所属行业', '采购方式', '采购人']:
        if col in row and not pd.isna(row[col]):
            texts.append(str(row[col]))
    return ' '.join(texts)


def balance_dataset(df, target_col='capacity_label', method='oversample'):
    """
    平衡数据集
    method: 'oversample' (上采样少数类) 或 'undersample' (下采样多数类)
    """
    print("\n  数据平衡处理...")

    # 获取各类别数量
    class_counts = df[target_col].value_counts()
    print(f"  原始分布: {dict(class_counts)}")

    if method == 'oversample':
        # 上采样到最大类的数量
        max_size = class_counts.max()
        dfs = []
        for label in class_counts.index:
            df_subset = df[df[target_col] == label]
            if len(df_subset) < max_size:
                df_upsampled = resample(df_subset,
                                        replace=True,
                                        n_samples=max_size,
                                        random_state=RANDOM_STATE)
                dfs.append(df_upsampled)
            else:
                dfs.append(df_subset)

        df_balanced = pd.concat(dfs)

    elif method == 'undersample':
        min_size = class_counts.min()
        dfs = []
        for label in class_counts.index:
            df_subset = df[df[target_col] == label]
            df_downsampled = resample(df_subset,
                                      replace=False,
                                      n_samples=min_size,
                                      random_state=RANDOM_STATE)
            dfs.append(df_downsampled)

        df_balanced = pd.concat(dfs)

    new_counts = df_balanced[target_col].value_counts()
    print(f"  平衡后分布: {dict(new_counts)}")

    return df_balanced.reset_index(drop=True)


# ============================================================================
# 第三部分：分类器
# ============================================================================

class ImprovedStateCapacityClassifier:
    """改进的分类器"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, df, test_size=0.2, balance=True):
        """准备数据"""
        print("\n" + "=" * 60)
        print("📊 准备训练数据...")
        print("=" * 60)

        # 创建文本特征
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        df['processed_text'] = df['combined_text'].apply(preprocess_text)

        # 数据平衡（仅对训练集）
        if balance:
            # 先分割
            X_train_df, X_test_df = train_test_split(
                df, test_size=test_size, random_state=RANDOM_STATE,
                stratify=df['capacity_label']
            )

            # 对训练集进行平衡
            X_train_df = balance_dataset(X_train_df, 'capacity_label', 'oversample')

            y_train = self.label_encoder.fit_transform(X_train_df['capacity_label'])
            y_test = self.label_encoder.transform(X_test_df['capacity_label'])

            X_train_text = X_train_df['processed_text'].values
            X_test_text = X_test_df['processed_text'].values

        else:
            y = self.label_encoder.fit_transform(df['capacity_label'])
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df['processed_text'].values, y,
                test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )

        print(f"\n  训练集: {len(X_train_text)} 条")
        print(f"  测试集: {len(X_test_text)} 条")
        print(f"  类别: {list(self.label_encoder.classes_)}")

        # TF-IDF特征
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )

        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        print(f"  特征维度: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """训练模型"""
        print("\n" + "=" * 60)
        print("🤖 训练机器学习模型...")
        print("=" * 60)

        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, C=1.0, solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=RANDOM_STATE
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale',
                random_state=RANDOM_STATE, probability=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, random_state=RANDOM_STATE
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=500, random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.1,
                activation='relu', solver='adam'
            )
        }

        results = []
        for name, model in model_configs.items():
            print(f"\n  训练 {name}...")

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            model.fit(X_train, y_train)
            self.models[name] = model

            result = {
                'model': name,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            results.append(result)
            print(f"    交叉验证 F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        best_result = max(results, key=lambda x: x['cv_mean'])
        self.best_model_name = best_result['model']
        self.best_model = self.models[self.best_model_name]

        print(f"\n  ✅ 最佳模型: {self.best_model_name}")
        return results

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n" + "=" * 60)
        print("📈 模型评估结果...")
        print("=" * 60)

        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'model': name,
                'accuracy': accuracy,
                'f1_macro': f1
            })
            print(f"\n  {name}: 准确率={accuracy:.4f}, F1={f1:.4f}")

        print(f"\n" + "=" * 60)
        print(f"📋 最佳模型 ({self.best_model_name}) 详细报告:")
        print("=" * 60)

        y_pred_best = self.best_model.predict(X_test)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_best,
                                    target_names=self.label_encoder.classes_))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred_best)
        print(pd.DataFrame(cm,
                           index=self.label_encoder.classes_,
                           columns=self.label_encoder.classes_))

        return results

    def predict(self, texts):
        """预测"""
        processed = [preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(processed)
        predictions = self.best_model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)

        if hasattr(self.best_model, 'predict_proba'):
            probs = self.best_model.predict_proba(X)
            return labels, probs

        return labels, None


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_results(df_labeled, eval_results, y_test, y_pred, label_encoder):
    """生成所有可视化图表"""

    print("\n" + "=" * 60)
    print("📊 生成可视化图表...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 标签分布
    label_counts = df_labeled['capacity_label'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0, 0].bar(label_counts.index, label_counts.values, color=colors)
    axes[0, 0].set_title('State Capacity Label Distribution', fontsize=12)
    axes[0, 0].set_xlabel('Capacity Type')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(label_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center')

    # 2. 模型对比
    df_results = pd.DataFrame(eval_results)
    x = np.arange(len(df_results))
    width = 0.35
    axes[0, 1].bar(x - width/2, df_results['accuracy'], width, label='Accuracy', color='#4ECDC4')
    axes[0, 1].bar(x + width/2, df_results['f1_macro'], width, label='F1-Macro', color='#FF6B6B')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Model Performance Comparison', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_results['model'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.0)

    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].figure.colorbar(im, ax=axes[1, 0])
    labels = label_encoder.classes_
    axes[1, 0].set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=labels, yticklabels=labels,
                   title='Confusion Matrix',
                   ylabel='True Label',
                   xlabel='Predicted Label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

    # 4. 置信度分布
    for label in df_labeled['capacity_label'].unique():
        subset = df_labeled[df_labeled['capacity_label'] == label]
        axes[1, 1].hist(subset['label_confidence'], bins=15, alpha=0.6, label=label)
    axes[1, 1].set_title('Label Confidence Distribution', fontsize=12)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('analysis_results_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  图表已保存: analysis_results_v2.png")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("  政府采购合同国家能力分类器 V2")
    print("  基于 Berwick & Christia (2018) 《State Capacity Redux》")
    print("=" * 70)

    # 1. 加载数据
    df = load_data()

    # 2. 改进的规则标注
    print("\n" + "=" * 60)
    print("🏷️  基于改进规则的初始标注...")
    print("=" * 60)

    labeler = ImprovedStateCapacityLabeler()
    df_labeled = labeler.label_dataframe(df)

    label_counts = df_labeled['capacity_label'].value_counts()
    print("\n  标签分布:")
    for label, count in label_counts.items():
        pct = count / len(df_labeled) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    # 3. 训练分类器
    classifier = ImprovedStateCapacityClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_labeled, balance=True)
    cv_results = classifier.train_models(X_train, y_train)
    eval_results = classifier.evaluate(X_test, y_test)

    # 4. 可视化
    y_pred = classifier.best_model.predict(X_test)
    plot_results(df_labeled, eval_results, y_test, y_pred, classifier.label_encoder)

    # 5. 示例预测
    print("\n" + "=" * 60)
    print("🔮 示例预测:")
    print("=" * 60)

    test_texts = [
        "税务系统升级改造项目 税务局",
        "XX小学教学设备采购 教育局",
        "市政道路维修工程 住建局",
        "医院CT设备采购 综合医院",
        "财政预算管理系统 财政局",
        "办公家具采购 政府办公室",
        "疫苗采购项目 疾控中心",
        "土地确权登记系统 国土局",
    ]

    labels, probs = classifier.predict(test_texts)

    label_names_cn = {
        'extractive': '汲取能力 (Extractive)',
        'coordination': '协调能力 (Coordination)',
        'compliance': '合规能力 (Compliance)'
    }

    for text, label in zip(test_texts, labels):
        print(f"  '{text[:30]}...' → {label_names_cn.get(label, label)}")

    # 6. 保存结果
    print("\n" + "=" * 60)
    print("💾 保存结果...")
    print("=" * 60)

    output_cols = ['年份', '合同名称', '主要标的名称', '采购人', '供应商',
                   '所属行业', '合同金额num_万元', 'capacity_label',
                   'label_confidence', 'label_reason']
    df_output = df_labeled[[c for c in output_cols if c in df_labeled.columns]]
    df_output.to_csv('classified_contracts_v2.csv', index=False, encoding='utf-8-sig')
    print("  分类结果已保存: classified_contracts_v2.csv")

    pd.DataFrame(eval_results).to_csv('model_evaluation_v2.csv', index=False)
    print("  模型评估已保存: model_evaluation_v2.csv")

    # 7. 生成分类摘要
    print("\n" + "=" * 70)
    print("📊 分类结果摘要")
    print("=" * 70)

    summary = df_labeled.groupby('capacity_label').agg({
        '合同金额num_万元': ['count', 'sum', 'mean'],
    }).round(2)
    summary.columns = ['合同数量', '金额总计(万元)', '平均金额(万元)']
    print(summary)

    print("\n" + "=" * 70)
    print("  ✅ 分类完成!")
    print("=" * 70)

    return df_labeled, classifier


if __name__ == '__main__':
    df_labeled, classifier = main()
