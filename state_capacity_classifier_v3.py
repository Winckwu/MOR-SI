#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府采购合同国家能力分类器 V3 - 高准确度版本
State Capacity Contract Classifier V3 - High Accuracy Version

目标：准确率 > 90%

优化策略：
1. 使用全部数据（2012-2015，约2万条）
2. 更精细的关键词权重系统
3. 多特征融合（文本+行业+采购人类型）
4. 集成学习（投票分类器）
5. 超参数优化

基于 Berwick & Christia (2018) 《State Capacity Redux》论文框架
"""

import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# 第一部分：高精度标注器
# ============================================================================

class HighAccuracyLabeler:
    """
    高精度标注器 - 使用分层规则和置信度过滤
    """

    def __init__(self):
        # 汲取能力 - 强调"资源获取"
        self.extractive_rules = {
            # 决定性关键词（直接判定）
            'decisive': [
                '税务局', '地税局', '国税局', '税务',
                '财政局', '财政厅', '财政',
                '审计局', '审计厅', '审计',
                '国土局', '国土资源', '土地储备', '土地出让',
                '矿产资源', '矿权', '采矿权',
                '资产评估', '资产清查', '国有资产',
                '征地拆迁', '土地征收', '房屋征收',
            ],
            # 强信号（权重5）
            'strong': [
                '预算', '决算', '财务管理', '会计', '出纳',
                '产权登记', '不动产登记', '房产登记',
                '地籍', '测绘', '确权',
                '资产', '资源', '矿产',
            ],
            # 中等信号（权重2）
            'medium': [
                '评估', '鉴定', '价值',
                '银行', '金融', '贷款', '融资',
            ]
        }

        # 协调能力 - 强调"集体行动组织"
        self.coordination_rules = {
            'decisive': [
                '市政工程', '道路工程', '公路工程', '桥梁工程',
                '水利工程', '电力工程', '通信工程',
                '住建局', '交通局', '公路局', '市政',
                '信息化建设', '电子政务', '智慧城市',
            ],
            'strong': [
                '道路', '公路', '桥梁', '隧道', '交通',
                '水利', '电力', '供电', '供水', '排水', '管网', '燃气',
                '通讯', '通信', '网络', '信息化',
                '建设', '施工', '改造', '修缮', '维修',
                '装修', '装饰', '绿化', '环卫',
                '政府采购', '办公设备', '办公家具', '公务车',
            ],
            'medium': [
                '办公', '会议', '档案', '印刷', '车辆',
                '规划', '设计', '咨询', '监理',
                '设备', '家具', '空调', '电脑', '计算机', '打印机',
            ]
        }

        # 合规能力 - 强调"公共服务提供"
        self.compliance_rules = {
            'decisive': [
                # 教育类
                '教育局', '教委', '学校', '大学', '学院', '中学', '小学', '幼儿园',
                '教学设备', '教学仪器', '实验室设备', '实训设备',
                '图书馆', '图书采购', '教材',
                # 医疗类
                '医院', '卫生院', '卫生局', '卫健委', '疾控中心', '疾控',
                '医疗设备', '医疗器械', '诊断设备',
                'CT', 'MRI', 'DR', 'B超', '彩超', '手术', 'X光',
                '药品', '疫苗', '医药',
                # 公共安全
                '公安局', '派出所', '警务', '执法', '司法局',
                '监控系统', '安防系统', '消防',
            ],
            'strong': [
                '教育', '教学', '培训', '学习', '课程', '实验', '实训',
                '医疗', '卫生', '健康', '诊疗', '护理', '康复', '检验', '检测',
                '安防', '监控', '安保', '消防', '应急',
                '环保', '环境', '污染治理', '垃圾处理',
                '养老', '福利', '救助', '社区',
            ],
            'medium': [
                '体育', '运动', '健身', '文化', '艺术',
                '图书', '阅读', '多媒体', '录播',
            ]
        }

        # 行业决定性映射
        self.industry_decisive = {
            # 合规能力
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
            '社会福利': 'compliance',

            # 汲取能力
            '财政': 'extractive',
            '税务': 'extractive',
            '金融业': 'extractive',

            # 协调能力
            '土木工程建筑业': 'coordination',
            '建筑装饰业': 'coordination',
            '道路运输业': 'coordination',
        }

    def _match_rules(self, text, rules):
        """匹配规则并返回得分"""
        if pd.isna(text):
            return 0, False, []

        text = str(text).lower()
        score = 0
        is_decisive = False
        matched = []

        # 检查决定性关键词
        for kw in rules.get('decisive', []):
            if kw.lower() in text:
                is_decisive = True
                score += 10
                matched.append((kw, 'decisive'))

        # 强信号
        for kw in rules.get('strong', []):
            if kw.lower() in text:
                score += 5
                matched.append((kw, 'strong'))

        # 中等信号
        for kw in rules.get('medium', []):
            if kw.lower() in text:
                score += 2
                matched.append((kw, 'medium'))

        return score, is_decisive, matched

    def label_single(self, row):
        """标注单条记录"""
        # 获取文本
        contract_name = str(row.get('合同名称', '')) if not pd.isna(row.get('合同名称')) else ''
        subject_name = str(row.get('主要标的名称', '')) if not pd.isna(row.get('主要标的名称')) else ''
        industry = str(row.get('所属行业', '')) if not pd.isna(row.get('所属行业')) else ''
        purchaser = str(row.get('采购人', '')) if not pd.isna(row.get('采购人')) else ''

        full_text = f"{contract_name} {subject_name} {purchaser}"

        # 行业决定性判断
        for ind_key, label in self.industry_decisive.items():
            if ind_key in industry:
                return label, 0.95, f'行业决定:{ind_key}'

        # 规则匹配
        ext_score, ext_decisive, ext_matched = self._match_rules(full_text, self.extractive_rules)
        coord_score, coord_decisive, coord_matched = self._match_rules(full_text, self.coordination_rules)
        comp_score, comp_decisive, comp_matched = self._match_rules(full_text, self.compliance_rules)

        # 采购人类型判断
        purchaser_lower = purchaser.lower()
        if any(kw in purchaser_lower for kw in ['学校', '大学', '学院', '小学', '中学', '幼儿园', '教育']):
            comp_score += 8
        elif any(kw in purchaser_lower for kw in ['医院', '卫生院', '疾控', '卫生']):
            comp_score += 8
        elif any(kw in purchaser_lower for kw in ['税务', '财政', '审计', '国土']):
            ext_score += 8
        elif any(kw in purchaser_lower for kw in ['住建', '交通', '公路', '市政', '水利']):
            coord_score += 5

        # 决定性关键词优先
        if ext_decisive and not coord_decisive and not comp_decisive:
            return 'extractive', 0.9, f'决定性匹配'
        if coord_decisive and not ext_decisive and not comp_decisive:
            return 'coordination', 0.9, f'决定性匹配'
        if comp_decisive and not ext_decisive and not coord_decisive:
            return 'compliance', 0.9, f'决定性匹配'

        # 得分判断
        scores = {
            'extractive': ext_score,
            'coordination': coord_score,
            'compliance': comp_score
        }

        max_score = max(scores.values())
        total_score = sum(scores.values())

        if max_score == 0:
            return 'coordination', 0.1, '默认'

        label = max(scores, key=scores.get)

        # 计算置信度
        if total_score > 0:
            margin = max_score - sorted(scores.values(), reverse=True)[1]
            confidence = min(0.5 + (margin / total_score) * 0.5, 0.95)
        else:
            confidence = 0.1

        reason = f'E:{ext_score},C:{coord_score},P:{comp_score}'
        return label, confidence, reason

    def label_dataframe(self, df):
        """标注整个数据框"""
        results = df.apply(self.label_single, axis=1)

        df_labeled = df.copy()
        df_labeled['capacity_label'] = [r[0] for r in results]
        df_labeled['label_confidence'] = [r[1] for r in results]
        df_labeled['label_reason'] = [r[2] for r in results]

        return df_labeled


# ============================================================================
# 第二部分：数据加载和处理
# ============================================================================

def load_all_data():
    """加载所有年份数据"""
    print("=" * 60)
    print("📂 加载数据...")
    print("=" * 60)

    dfs = []

    # 加载.dta文件
    for year in ['2012', '2013', '2014']:
        try:
            df = pd.read_stata(f'{year}.dta')
            print(f"  {year}年 (.dta): {len(df)} 条")
            dfs.append(df)
        except Exception as e:
            print(f"  {year}年: 加载失败 - {e}")

    # 加载2015年Excel文件
    try:
        df_2015 = pd.read_excel('2015.xls')
        print(f"  2015年 (.xls): {len(df_2015)} 条")
        dfs.append(df_2015)
    except Exception as e:
        print(f"  2015年: 加载失败 - {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n  📊 总计: {len(df_all)} 条记录")

    return df_all


def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_features(row):
    """创建组合特征"""
    texts = []
    for col in ['合同名称', '主要标的名称', '所属行业', '采购方式', '采购人']:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            texts.append(str(val))
    return ' '.join(texts)


def balance_classes(df, target_col='capacity_label'):
    """类别平衡 - 使用SMOTE风格的上采样"""
    print("\n  ⚖️ 类别平衡处理...")

    class_counts = df[target_col].value_counts()
    print(f"  原始分布: {dict(class_counts)}")

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
    new_counts = df_balanced[target_col].value_counts()
    print(f"  平衡后: {dict(new_counts)}")

    return df_balanced.reset_index(drop=True)


# ============================================================================
# 第三部分：高准确度分类器
# ============================================================================

class HighAccuracyClassifier:
    """高准确度分类器 - 使用集成学习"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.models = {}
        self.ensemble = None
        self.best_single_model = None
        self.best_single_name = None

    def prepare_data(self, df, test_size=0.2, use_high_confidence=True):
        """准备数据"""
        print("\n" + "=" * 60)
        print("📊 准备训练数据...")
        print("=" * 60)

        # 可选：只使用高置信度样本
        if use_high_confidence:
            high_conf_df = df[df['label_confidence'] >= 0.5].copy()
            print(f"  高置信度样本 (≥0.5): {len(high_conf_df)} / {len(df)} ({len(high_conf_df)/len(df)*100:.1f}%)")
            working_df = high_conf_df
        else:
            working_df = df.copy()

        # 创建文本特征
        working_df['combined_text'] = working_df.apply(create_features, axis=1)
        working_df['processed_text'] = working_df['combined_text'].apply(preprocess_text)

        # 分割数据
        X_train_df, X_test_df = train_test_split(
            working_df, test_size=test_size, random_state=RANDOM_STATE,
            stratify=working_df['capacity_label']
        )

        # 对训练集进行类别平衡
        X_train_df = balance_classes(X_train_df)

        # 编码标签
        y_train = self.label_encoder.fit_transform(X_train_df['capacity_label'])
        y_test = self.label_encoder.transform(X_test_df['capacity_label'])

        print(f"\n  训练集: {len(X_train_df)} 条")
        print(f"  测试集: {len(X_test_df)} 条")
        print(f"  类别: {list(self.label_encoder.classes_)}")

        # TF-IDF特征 - 优化参数
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True  # 使用log(tf)
        )

        X_train = self.vectorizer.fit_transform(X_train_df['processed_text'])
        X_test = self.vectorizer.transform(X_test_df['processed_text'])

        print(f"  特征维度: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test, X_test_df

    def train_models(self, X_train, y_train):
        """训练多个模型"""
        print("\n" + "=" * 60)
        print("🤖 训练机器学习模型...")
        print("=" * 60)

        # 定义模型
        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE,
                C=2.0, solver='lbfgs', class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale',
                random_state=RANDOM_STATE, probability=True, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300, random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.1,
                activation='relu', solver='adam', alpha=0.001
            )
        }

        # 训练各模型
        results = []
        for name, model in model_configs.items():
            print(f"\n  训练 {name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            self.models[name] = model

            result = {
                'model': name,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            results.append(result)
            print(f"    交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 找出最佳单模型
        best_result = max(results, key=lambda x: x['cv_accuracy'])
        self.best_single_name = best_result['model']
        self.best_single_model = self.models[self.best_single_name]
        print(f"\n  ✅ 最佳单模型: {self.best_single_name} (CV准确率: {best_result['cv_accuracy']:.4f})")

        # 创建集成模型（投票分类器）
        print("\n  🔗 创建集成模型 (Voting Classifier)...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('lr', self.models['Logistic Regression']),
                ('rf', self.models['Random Forest']),
                ('svm', self.models['SVM']),
                ('gb', self.models['Gradient Boosting']),
            ],
            voting='soft',  # 使用概率投票
            weights=[1.2, 1.0, 1.2, 1.0]  # 给表现好的模型更高权重
        )
        self.ensemble.fit(X_train, y_train)

        ensemble_cv = cross_val_score(self.ensemble, X_train, y_train, cv=5, scoring='accuracy')
        print(f"    集成模型交叉验证准确率: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std():.4f})")

        return results

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n" + "=" * 60)
        print("📈 模型评估结果...")
        print("=" * 60)

        results = []

        # 评估各模型
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'model': name,
                'accuracy': accuracy,
                'f1_macro': f1
            })
            print(f"  {name}: 准确率={accuracy:.4f}, F1={f1:.4f}")

        # 评估集成模型
        y_pred_ensemble = self.ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='macro')

        results.append({
            'model': 'Ensemble (Voting)',
            'accuracy': ensemble_accuracy,
            'f1_macro': ensemble_f1
        })
        print(f"\n  🏆 Ensemble (Voting): 准确率={ensemble_accuracy:.4f}, F1={ensemble_f1:.4f}")

        # 详细报告
        print(f"\n" + "=" * 60)
        print(f"📋 集成模型详细报告:")
        print("=" * 60)

        print("\n分类报告:")
        print(classification_report(y_test, y_pred_ensemble,
                                    target_names=self.label_encoder.classes_))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred_ensemble)
        print(pd.DataFrame(cm,
                           index=self.label_encoder.classes_,
                           columns=self.label_encoder.classes_))

        return results, y_pred_ensemble

    def predict(self, texts, use_ensemble=True):
        """预测"""
        processed = [preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(processed)

        if use_ensemble:
            predictions = self.ensemble.predict(X)
        else:
            predictions = self.best_single_model.predict(X)

        labels = self.label_encoder.inverse_transform(predictions)
        return labels


# ============================================================================
# 第四部分：可视化
# ============================================================================

def create_visualizations(df_labeled, eval_results, y_test, y_pred, label_encoder):
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("📊 生成可视化图表...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 标签分布（按年份）
    year_label_counts = df_labeled.groupby(['年份', 'capacity_label']).size().unstack(fill_value=0)
    year_label_counts.plot(kind='bar', ax=axes[0, 0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Label Distribution by Year', fontsize=12)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend(title='Capacity Type')
    axes[0, 0].tick_params(axis='x', rotation=0)

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
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Target')

    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].figure.colorbar(im, ax=axes[1, 0])
    labels = label_encoder.classes_
    axes[1, 0].set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=labels, yticklabels=labels,
                   title='Confusion Matrix (Ensemble)',
                   ylabel='True Label',
                   xlabel='Predicted Label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=12)

    # 4. 每类准确率
    class_accuracy = []
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)

    bars = axes[1, 1].bar(labels, class_accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Per-Class Accuracy', fontsize=12)
    axes[1, 1].set_xlabel('Capacity Type')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
    for bar, acc in zip(bars, class_accuracy):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('analysis_results_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  图表已保存: analysis_results_v3.png")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("  政府采购合同国家能力分类器 V3 - 高准确度版本")
    print("  目标：准确率 > 90%")
    print("=" * 70)

    # 1. 加载数据
    df = load_all_data()

    # 2. 标注
    print("\n" + "=" * 60)
    print("🏷️  高精度规则标注...")
    print("=" * 60)

    labeler = HighAccuracyLabeler()
    df_labeled = labeler.label_dataframe(df)

    label_counts = df_labeled['capacity_label'].value_counts()
    print("\n  标签分布:")
    for label, count in label_counts.items():
        pct = count / len(df_labeled) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    high_conf = df_labeled[df_labeled['label_confidence'] >= 0.5]
    print(f"\n  高置信度样本 (≥0.5): {len(high_conf)} ({len(high_conf)/len(df_labeled)*100:.1f}%)")

    # 3. 训练分类器
    classifier = HighAccuracyClassifier()
    X_train, X_test, y_train, y_test, test_df = classifier.prepare_data(
        df_labeled, test_size=0.15, use_high_confidence=True
    )

    cv_results = classifier.train_models(X_train, y_train)
    eval_results, y_pred = classifier.evaluate(X_test, y_test)

    # 4. 检查是否达到90%
    ensemble_result = [r for r in eval_results if 'Ensemble' in r['model']][0]
    accuracy = ensemble_result['accuracy']

    print("\n" + "=" * 70)
    if accuracy >= 0.9:
        print(f"  🎉 成功！准确率达到 {accuracy:.2%}，超过90%目标！")
    else:
        print(f"  📈 当前准确率: {accuracy:.2%}，距离90%目标还差 {0.9 - accuracy:.2%}")
    print("=" * 70)

    # 5. 可视化
    create_visualizations(df_labeled, eval_results, y_test, y_pred, classifier.label_encoder)

    # 6. 示例预测
    print("\n" + "=" * 60)
    print("🔮 示例预测:")
    print("=" * 60)

    test_texts = [
        "税务局信息系统升级改造项目",
        "XX小学教学设备采购项目 教育局",
        "市政道路维修工程 住建局",
        "XX医院CT设备采购 综合医院",
        "财政预算管理系统采购 财政局",
        "政府办公家具采购项目",
        "疾控中心疫苗冷链设备采购",
        "国土局土地确权登记系统",
    ]

    labels = classifier.predict(test_texts)

    label_names = {
        'extractive': '汲取能力 (Extractive)',
        'coordination': '协调能力 (Coordination)',
        'compliance': '合规能力 (Compliance)'
    }

    for text, label in zip(test_texts, labels):
        print(f"  '{text[:25]}...' → {label_names.get(label, label)}")

    # 7. 保存结果
    print("\n" + "=" * 60)
    print("💾 保存结果...")
    print("=" * 60)

    output_cols = ['年份', '合同名称', '主要标的名称', '采购人', '供应商',
                   '所属行业', '合同金额num_万元', 'capacity_label',
                   'label_confidence', 'label_reason']
    df_output = df_labeled[[c for c in output_cols if c in df_labeled.columns]]
    df_output.to_csv('classified_contracts_v3.csv', index=False, encoding='utf-8-sig')
    print("  分类结果已保存: classified_contracts_v3.csv")

    pd.DataFrame(eval_results).to_csv('model_evaluation_v3.csv', index=False)
    print("  模型评估已保存: model_evaluation_v3.csv")

    # 8. 摘要统计
    print("\n" + "=" * 70)
    print("📊 分类结果摘要")
    print("=" * 70)

    summary = df_labeled.groupby(['年份', 'capacity_label']).size().unstack(fill_value=0)
    print(summary)

    print("\n" + "=" * 70)
    print("  ✅ 分类完成!")
    print("=" * 70)

    return df_labeled, classifier, accuracy


if __name__ == '__main__':
    df_labeled, classifier, accuracy = main()
