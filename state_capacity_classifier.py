#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府采购合同国家能力分类器
State Capacity Contract Classifier

基于 Berwick & Christia (2018) 《State Capacity Redux》论文框架
将政府采购合同按照三种国家能力进行分类：
- 汲取能力 (Extractive Capacity): 国家获取资源的能力
- 协调能力 (Coordination Capacity): 组织集体行动的能力
- 合规能力 (Compliance Capacity): 确保服从的能力

作者: Claude AI
日期: 2025-11-28
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 第一部分：分类规则定义
# ============================================================================

class StateCapacityLabeler:
    """
    基于论文框架的规则标注器

    根据 Berwick & Christia (2018) 的理论框架：

    1. 汲取能力 (Extractive):
       - 核心是统治者与资源持有者之间的关系
       - 涉及税收、财政、资源获取、审计等
       - 关键指标：税收比率、财政收入

    2. 协调能力 (Coordination):
       - 依赖官僚与社会成员的关系
       - 涉及基础设施、公共服务协调、官僚体系运作
       - 韦伯式专业官僚制的制度表现

    3. 合规能力 (Compliance):
       - 确保公民、精英和官僚服从国家目标
       - 主要涉及教育、医疗等公共服务提供
       - 官僚选拔、激励和监督机制
    """

    def __init__(self):
        # 汲取能力关键词（与资源获取、税收、财政相关）
        self.extractive_keywords = [
            # 财税相关
            '税务', '税收', '财税', '财政', '预算', '审计', '会计',
            '财务', '资金', '收费', '缴费', '罚款', '罚没',
            # 资产资源相关
            '资产', '资源', '矿产', '国土', '土地', '房产', '不动产',
            '登记', '产权', '确权', '地籍', '测绘',
            # 征收相关
            '征收', '拆迁', '补偿', '评估',
            # 金融相关
            '银行', '金融', '贷款', '融资',
        ]

        # 协调能力关键词（与基础设施、行政协调相关）
        self.coordination_keywords = [
            # 基础设施
            '道路', '公路', '桥梁', '隧道', '交通', '运输',
            '水利', '电力', '供电', '供水', '排水', '管网', '燃气',
            '通讯', '通信', '网络', '信息化', '电子政务', '数字化',
            # 建设工程
            '建设', '工程', '施工', '改造', '修缮', '维修',
            '装修', '装饰', '绿化', '环卫', '清洁',
            # 办公协调
            '办公', '行政', '会议', '档案', '印刷', '打印',
            '复印', '设备', '家具', '车辆', '后勤',
            # 规划管理
            '规划', '设计', '咨询', '监理', '管理',
        ]

        # 合规能力关键词（与公共服务提供、监管执法相关）
        self.compliance_keywords = [
            # 教育服务
            '教育', '教学', '学校', '培训', '课程', '教材', '教具',
            '实验', '实训', '录播', '多媒体', '图书', '阅读',
            '体育', '运动', '健身',
            # 医疗卫生
            '医疗', '医院', '卫生', '健康', '疾控', '防疫',
            '诊断', '治疗', '护理', '康复', '药品', '医药',
            '检验', '检测', '化验', '影像', 'CT', 'B超',
            # 公共安全
            '安防', '监控', '安保', '消防', '应急', '救援',
            '公安', '警务', '执法', '司法', '法院', '检察',
            # 社会服务
            '养老', '福利', '救助', '社区', '民政',
            '环保', '环境', '污染', '垃圾', '处理',
        ]

        # 行业到能力类型的映射
        self.industry_mapping = {
            # 汲取能力相关行业
            '金融业': 'extractive',
            '财政': 'extractive',
            '税务': 'extractive',

            # 协调能力相关行业
            '建筑业': 'coordination',
            '交通运输业': 'coordination',
            '邮政业': 'coordination',
            '信息传输': 'coordination',
            '房地产业': 'coordination',
            '租赁和商务服务业': 'coordination',

            # 合规能力相关行业
            '普通高等教育': 'compliance',
            '中等职业学校教育': 'compliance',
            '普通小学教育': 'compliance',
            '普通初中教育': 'compliance',
            '学前教育': 'compliance',
            '特殊教育': 'compliance',
            '教育': 'compliance',
            '综合医院': 'compliance',
            '专科医院': 'compliance',
            '卫生': 'compliance',
            '医疗': 'compliance',
            '社会工作': 'compliance',
            '公共管理': 'compliance',
        }

    def _count_keywords(self, text, keywords):
        """统计文本中关键词出现次数"""
        if pd.isna(text):
            return 0
        text = str(text).lower()
        count = 0
        for keyword in keywords:
            count += len(re.findall(keyword.lower(), text))
        return count

    def label_single(self, contract_name, subject_name=None, industry=None):
        """
        对单条记录进行标注

        返回: (标签, 置信度, 原因)
        """
        # 合并文本
        text = str(contract_name) if contract_name else ''
        if subject_name and not pd.isna(subject_name):
            text += ' ' + str(subject_name)

        # 统计各类关键词
        extractive_score = self._count_keywords(text, self.extractive_keywords)
        coordination_score = self._count_keywords(text, self.coordination_keywords)
        compliance_score = self._count_keywords(text, self.compliance_keywords)

        # 行业加权
        industry_bonus = 0
        industry_label = None
        if industry and not pd.isna(industry):
            for ind_key, label in self.industry_mapping.items():
                if ind_key in str(industry):
                    industry_label = label
                    industry_bonus = 2
                    break

        # 计算最终得分
        scores = {
            'extractive': extractive_score + (industry_bonus if industry_label == 'extractive' else 0),
            'coordination': coordination_score + (industry_bonus if industry_label == 'coordination' else 0),
            'compliance': compliance_score + (industry_bonus if industry_label == 'compliance' else 0)
        }

        # 确定标签
        max_score = max(scores.values())
        total_score = sum(scores.values())

        if max_score == 0:
            # 无明显特征，根据行业判断
            if industry_label:
                return industry_label, 0.3, '仅行业匹配'
            else:
                return 'coordination', 0.1, '默认分类（通用政府采购）'

        # 找出最高分的类别
        label = max(scores, key=scores.get)
        confidence = max_score / (total_score + 1)

        return label, confidence, f'关键词匹配(E:{extractive_score},C:{coordination_score},P:{compliance_score})'

    def label_dataframe(self, df):
        """对整个数据框进行标注"""
        labels = []
        confidences = []
        reasons = []

        for idx, row in df.iterrows():
            label, conf, reason = self.label_single(
                row.get('合同名称', ''),
                row.get('主要标的名称', ''),
                row.get('所属行业', '')
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
# 第二部分：数据加载和预处理
# ============================================================================

def load_data():
    """加载所有年份的数据"""
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
    # 去除特殊字符，保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_combined_text(row):
    """创建组合文本特征"""
    texts = []
    for col in ['合同名称', '主要标的名称', '所属行业', '采购方式']:
        if col in row and not pd.isna(row[col]):
            texts.append(str(row[col]))
    return ' '.join(texts)


# ============================================================================
# 第三部分：特征工程
# ============================================================================

class FeatureExtractor:
    """特征提取器"""

    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit_transform(self, texts):
        """训练并转换文本"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )

        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """转换文本"""
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """获取特征名称"""
        return self.vectorizer.get_feature_names_out()


# ============================================================================
# 第四部分：模型训练和评估
# ============================================================================

class StateCapacityClassifier:
    """国家能力分类器"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, df, test_size=0.2):
        """准备训练和测试数据"""
        print("\n" + "=" * 60)
        print("📊 准备训练数据...")
        print("=" * 60)

        # 创建组合文本
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        df['processed_text'] = df['combined_text'].apply(preprocess_text)

        # 编码标签
        y = self.label_encoder.fit_transform(df['capacity_label'])

        # 分割数据
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['processed_text'].values,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y
        )

        print(f"  训练集: {len(X_train_text)} 条")
        print(f"  测试集: {len(X_test_text)} 条")
        print(f"  类别分布:")
        for i, label in enumerate(self.label_encoder.classes_):
            train_count = sum(y_train == i)
            test_count = sum(y_test == i)
            print(f"    {label}: 训练={train_count}, 测试={test_count}")

        # 特征提取
        self.feature_extractor = FeatureExtractor(method='tfidf', max_features=3000)
        X_train = self.feature_extractor.fit_transform(X_train_text)
        X_test = self.feature_extractor.transform(X_test_text)

        print(f"\n  特征维度: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """训练多个模型"""
        print("\n" + "=" * 60)
        print("🤖 训练机器学习模型...")
        print("=" * 60)

        # 定义模型
        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='linear',
                random_state=RANDOM_STATE,
                class_weight='balanced',
                probability=True
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True
            )
        }

        # 训练每个模型
        results = []
        for name, model in model_configs.items():
            print(f"\n  训练 {name}...")

            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

            # 完整训练
            model.fit(X_train, y_train)
            self.models[name] = model

            result = {
                'model': name,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            results.append(result)
            print(f"    交叉验证 F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 找出最佳模型
        best_result = max(results, key=lambda x: x['cv_mean'])
        self.best_model_name = best_result['model']
        self.best_model = self.models[self.best_model_name]

        print(f"\n  ✅ 最佳模型: {self.best_model_name}")

        return results

    def evaluate(self, X_test, y_test):
        """评估所有模型"""
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

            print(f"\n  {name}:")
            print(f"    准确率: {accuracy:.4f}")
            print(f"    F1-Macro: {f1:.4f}")

        # 最佳模型详细报告
        print(f"\n" + "=" * 60)
        print(f"📋 最佳模型 ({self.best_model_name}) 详细报告:")
        print("=" * 60)

        y_pred_best = self.best_model.predict(X_test)
        print("\n分类报告:")
        print(classification_report(
            y_test,
            y_pred_best,
            target_names=self.label_encoder.classes_
        ))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred_best)
        print(pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        ))

        return results

    def predict(self, texts):
        """预测新文本"""
        processed = [preprocess_text(t) for t in texts]
        X = self.feature_extractor.transform(processed)
        predictions = self.best_model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)

        # 如果模型支持概率预测
        if hasattr(self.best_model, 'predict_proba'):
            probs = self.best_model.predict_proba(X)
            return labels, probs

        return labels, None

    def get_important_features(self, top_n=20):
        """获取重要特征"""
        feature_names = self.feature_extractor.get_feature_names()

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # 对于多分类，取各类别系数的平均绝对值
            importances = np.abs(self.best_model.coef_).mean(axis=0)
        else:
            return None

        # 排序
        indices = np.argsort(importances)[::-1][:top_n]

        return [(feature_names[i], importances[i]) for i in indices]


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_label_distribution(df, save_path='label_distribution.png'):
    """绘制标签分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 标签分布
    label_counts = df['capacity_label'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    axes[0].bar(label_counts.index, label_counts.values, color=colors)
    axes[0].set_title('国家能力类型分布', fontsize=14)
    axes[0].set_xlabel('能力类型')
    axes[0].set_ylabel('合同数量')

    # 添加数值标签
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=12)

    # 置信度分布
    for label in df['capacity_label'].unique():
        subset = df[df['capacity_label'] == label]
        axes[1].hist(subset['label_confidence'], bins=20, alpha=0.5, label=label)

    axes[1].set_title('标注置信度分布', fontsize=14)
    axes[1].set_xlabel('置信度')
    axes[1].set_ylabel('数量')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


def plot_model_comparison(results, save_path='model_comparison.png'):
    """绘制模型对比图"""
    df_results = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df_results))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_results['accuracy'], width, label='准确率', color='#4ECDC4')
    bars2 = ax.bar(x + width/2, df_results['f1_macro'], width, label='F1-Macro', color='#FF6B6B')

    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')

    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


# ============================================================================
# 第六部分：主程序
# ============================================================================

def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("  政府采购合同国家能力分类器")
    print("  基于 Berwick & Christia (2018) 《State Capacity Redux》")
    print("=" * 70)

    # 1. 加载数据
    df = load_data()

    # 2. 使用规则进行初始标注（生成训练数据）
    print("\n" + "=" * 60)
    print("🏷️  基于规则的初始标注...")
    print("=" * 60)

    labeler = StateCapacityLabeler()
    df_labeled = labeler.label_dataframe(df)

    # 标签统计
    print("\n  标签分布:")
    label_counts = df_labeled['capacity_label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df_labeled) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    # 高置信度样本统计
    high_conf = df_labeled[df_labeled['label_confidence'] >= 0.3]
    print(f"\n  高置信度样本 (≥0.3): {len(high_conf)} ({len(high_conf)/len(df_labeled)*100:.1f}%)")

    # 3. 可视化标签分布
    print("\n" + "=" * 60)
    print("📊 生成可视化图表...")
    print("=" * 60)
    plot_label_distribution(df_labeled)

    # 4. 训练机器学习模型
    classifier = StateCapacityClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_labeled)

    # 5. 训练模型
    cv_results = classifier.train_models(X_train, y_train)

    # 6. 评估模型
    eval_results = classifier.evaluate(X_test, y_test)

    # 7. 绘制模型对比图
    plot_model_comparison(eval_results)

    # 8. 绘制混淆矩阵
    y_pred = classifier.best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, classifier.label_encoder.classes_)

    # 9. 显示重要特征
    print("\n" + "=" * 60)
    print("🔑 重要特征 (Top 20):")
    print("=" * 60)
    important_features = classifier.get_important_features(top_n=20)
    if important_features:
        for i, (feature, importance) in enumerate(important_features, 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")

    # 10. 保存结果
    print("\n" + "=" * 60)
    print("💾 保存结果...")
    print("=" * 60)

    # 保存标注后的数据
    output_cols = ['年份', '合同名称', '主要标的名称', '采购人', '供应商',
                   '所属行业', '合同金额num_万元', 'capacity_label',
                   'label_confidence', 'label_reason']
    df_output = df_labeled[[c for c in output_cols if c in df_labeled.columns]]
    df_output.to_csv('classified_contracts.csv', index=False, encoding='utf-8-sig')
    print("  分类结果已保存: classified_contracts.csv")

    # 保存模型评估结果
    pd.DataFrame(eval_results).to_csv('model_evaluation.csv', index=False)
    print("  模型评估已保存: model_evaluation.csv")

    # 11. 示例预测
    print("\n" + "=" * 60)
    print("🔮 示例预测:")
    print("=" * 60)

    test_texts = [
        "税务系统升级改造项目",
        "小学教学设备采购",
        "市政道路维修工程",
        "医院医疗器械采购",
        "财政预算管理系统",
        "办公家具采购"
    ]

    labels, probs = classifier.predict(test_texts)

    label_names_cn = {
        'extractive': '汲取能力',
        'coordination': '协调能力',
        'compliance': '合规能力'
    }

    for text, label in zip(test_texts, labels):
        print(f"  '{text}' → {label_names_cn.get(label, label)}")

    print("\n" + "=" * 70)
    print("  ✅ 分类完成!")
    print("=" * 70)

    return df_labeled, classifier


if __name__ == '__main__':
    df_labeled, classifier = main()
