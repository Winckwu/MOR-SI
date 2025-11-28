#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器
基于 Berwick & Christia (2018) "State Capacity Redux" 论文框架

三种国家能力分类：
1. 汲取能力 (Extractive Capacity) - 国家获取资源的能力
2. 协调能力 (Coordination Capacity) - 国家组织集体行动的能力
3. 合规能力 (Compliance Capacity) - 确保服从国家目标的能力

作者：基于学术文献的关键词体系
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 尝试导入jieba，如果失败则使用简单分词
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("警告: jieba未安装，使用字符级分词")


def simple_tokenizer(text):
    """简单的中文分词器（字符级 + bigram）"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    # 字符级分词
    chars = list(text)
    # 添加bigram
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


def tokenize(text):
    """分词函数"""
    if HAS_JIEBA:
        return list(jieba.cut(str(text)))
    else:
        return simple_tokenizer(text)

# ============================================================
# 第一部分：基于论文的关键词词典构建
# 所有关键词均有明确的学术文献来源
# ============================================================

KEYWORD_DICTIONARY = {
    "汲取能力": {
        "description": "国家获取资源的能力 (Extractive Capacity)",
        "theoretical_sources": [
            "Tilly (1990) - 战争驱动财政发展",
            "Levi (1988) - 收入最大化假设",
            "Besley & Persson (2014) - 税收能力研究",
            "D'Arcy & Nistotskaya (2016) - 地籍调查与国家能力",
            "Brambor et al. (2016) - 统计能力与国家能力",
            "North & Weingast (1989) - 产权保护与可信承诺"
        ],
        "keywords": {
            # 核心关键词 - 直接来自论文引用的概念
            "税收财政类": ["税", "税收", "税务", "财税", "财政", "国税", "地税",
                       "征收", "缴纳", "纳税", "税源", "税基"],
            "预算收入类": ["预算", "决算", "收入", "岁入", "资金", "经费", "拨款"],
            "土地资源类": ["土地调查", "地籍", "确权", "不动产登记", "土地登记",
                       "土地变更", "耕地", "宅基地", "国土"],
            "统计普查类": ["统计", "普查", "调查", "登记", "年鉴", "数据采集"],
            "资产评估类": ["资产", "评估", "估价", "清产核资", "产权"],
            "审计监管类": ["审计", "稽查", "核查", "查账"],  # 财务审计属于汲取能力
            "关税贸易类": ["关税", "海关", "进出口", "口岸", "边检"]
        }
    },

    "协调能力": {
        "description": "国家组织集体行动的能力 (Coordination Capacity)",
        "theoretical_sources": [
            "Mann (1984, 2008) - 基础设施权力",
            "Weber (1946) - 官僚制与行政效率",
            "Evans (1989) - 嵌入式自主与发展型国家",
            "Johnson (1982) - 产业政策与经济协调",
            "Polanyi (1944) - 市场监管与社会保护",
            "Herbst (2000) - 地理渗透与道路网络",
            "Andrews et al. (2017) - 地方能力建设"
        ],
        "keywords": {
            # 基础设施类 - 来自Mann的infrastructural power概念
            "交通基础设施": ["公路", "铁路", "高速", "道路", "桥梁", "隧道",
                        "轨道交通", "地铁", "机场", "港口", "码头", "航道"],
            "市政基础设施": ["水利", "供水", "排水", "污水", "给水", "自来水",
                        "电力", "供电", "电网", "变电", "输电",
                        "燃气", "天然气", "供暖", "供热", "管网"],
            "信息基础设施": ["通信", "网络", "信息化", "数字化", "宽带",
                        "光纤", "基站", "覆盖", "信号"],
            # 行政管理类 - 来自Weber的bureaucracy概念
            "行政办公类": ["行政", "办公", "机关", "政务", "政府"],
            # 规划建设类 - 来自Evans的developmental state概念
            "规划发展类": ["规划", "建设", "工程", "项目", "发展", "改造",
                       "重建", "新建", "扩建", "城镇化", "城市化"],
            # 产业经济类 - 来自Johnson的industrial policy概念
            "产业园区类": ["产业", "园区", "开发区", "经济区", "工业"],
            # 标准规范类 - 来自Polanyi的市场监管概念
            "标准检测类": ["标准", "规范", "检测", "检验", "认证", "质量",
                       "计量", "校准", "鉴定"]
        }
    },

    "合规能力": {
        "description": "确保服从国家目标的能力 (Compliance Capacity)",
        "theoretical_sources": [
            "Olken (2007) - 监督与反腐实验",
            "Bjorkman & Svensson (2009) - 社区监督与医疗服务",
            "Muralidharan & Sundararaman (2011) - 教师激励与教育质量",
            "Banerjee et al. (2010) - 公民参与与教育",
            "Basinga et al. (2011) - 绩效激励与医疗服务",
            "Holland (2015) - 执法选择性",
            "Tsai (2007) - 非正式问责机制",
            "Ferraz & Finan (2008) - 审计与政治问责"
        ],
        "keywords": {
            # 监督检查类 - 来自Olken的monitoring实验
            "监督监控类": ["监督", "监控", "监察", "巡视", "巡查", "督查",
                       "检查", "抽查", "核查", "视察"],
            # 考核评估类 - 来自Muralidharan的绩效研究
            "考核培训类": ["考核", "考试", "评估", "评价", "绩效", "奖惩",
                       "培训", "教育培训", "业务培训", "技能培训"],
            # 公共服务类 - 来自Bjorkman & Svensson的服务提供研究
            "教育服务类": ["教育", "学校", "教学", "课堂", "教室", "师资",
                       "幼儿园", "中学", "小学", "大学", "职业教育",
                       "改薄", "义务教育", "教体"],
            "医疗卫生类": ["医疗", "卫生", "健康", "医院", "诊所", "疾控",
                       "防疫", "疫苗", "疫情", "免疫", "药品", "医药",
                       "计生", "妇幼", "康复"],
            "社会保障类": ["社保", "养老", "低保", "救助", "扶贫", "民政",
                       "残疾", "福利", "殡葬", "救灾", "应急"],
            # 执法强制类 - 来自Holland的enforcement研究
            "执法处罚类": ["执法", "处罚", "惩戒", "强制", "取缔", "查处",
                       "公安", "警察", "警务", "消防", "安防", "监狱"],
            # 问责反馈类 - 来自Tsai的accountability研究
            "问责投诉类": ["问责", "责任", "投诉", "举报", "信访", "申诉"]
        }
    }
}


def get_all_keywords():
    """获取所有关键词及其对应的能力类型"""
    keyword_to_capacity = {}
    for capacity_type, content in KEYWORD_DICTIONARY.items():
        for category, keywords in content["keywords"].items():
            for keyword in keywords:
                keyword_to_capacity[keyword] = {
                    "capacity": capacity_type,
                    "category": category
                }
    return keyword_to_capacity


def print_keyword_sources():
    """打印关键词的学术来源"""
    print("=" * 70)
    print("关键词理论来源说明")
    print("基于 Berwick & Christia (2018) 'State Capacity Redux' 论文框架")
    print("=" * 70)

    for capacity_type, content in KEYWORD_DICTIONARY.items():
        print(f"\n【{capacity_type}】")
        print(f"定义: {content['description']}")
        print("理论来源:")
        for source in content["theoretical_sources"]:
            print(f"  - {source}")
        print("关键词类别:")
        for category, keywords in content["keywords"].items():
            print(f"  {category}: {', '.join(keywords[:5])}...")


# ============================================================
# 第二部分：基于规则的自动标注系统
# ============================================================

class RuleBasedLabeler:
    """基于关键词规则的标注器"""

    def __init__(self):
        self.keyword_dict = get_all_keywords()
        self.all_keywords = list(self.keyword_dict.keys())
        # 按长度降序排列，优先匹配长关键词
        self.all_keywords.sort(key=len, reverse=True)

    def find_keywords(self, text):
        """在文本中查找所有匹配的关键词"""
        if pd.isna(text):
            return []

        found = []
        for keyword in self.all_keywords:
            if keyword in str(text):
                found.append({
                    "keyword": keyword,
                    "capacity": self.keyword_dict[keyword]["capacity"],
                    "category": self.keyword_dict[keyword]["category"]
                })
        return found

    def label_contract(self, contract_name):
        """
        对单个合同进行标注
        返回: (标签, 置信度, 匹配的关键词)
        """
        if pd.isna(contract_name):
            return None, 0, []

        found_keywords = self.find_keywords(contract_name)

        if not found_keywords:
            return None, 0, []

        # 统计每种能力类型的关键词数量
        capacity_counts = Counter([kw["capacity"] for kw in found_keywords])

        # 计算权重分数（考虑关键词长度作为权重）
        capacity_scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}
        for kw in found_keywords:
            # 关键词越长，权重越高
            weight = len(kw["keyword"])
            capacity_scores[kw["capacity"]] += weight

        # 找出得分最高的能力类型
        max_capacity = max(capacity_scores, key=capacity_scores.get)
        max_score = capacity_scores[max_capacity]
        total_score = sum(capacity_scores.values())

        if total_score == 0:
            return None, 0, []

        # 计算置信度
        confidence = max_score / total_score

        # 只有当置信度较高时才标注
        if confidence >= 0.5 and max_score >= 2:  # 至少匹配到长度>=2的关键词
            return max_capacity, confidence, found_keywords
        else:
            return None, confidence, found_keywords

    def label_dataset(self, df, text_column="合同名称"):
        """对整个数据集进行标注"""
        results = []

        for idx, row in df.iterrows():
            text = row[text_column]
            label, confidence, keywords = self.label_contract(text)

            results.append({
                "index": idx,
                "text": text,
                "label": label,
                "confidence": confidence,
                "matched_keywords": [kw["keyword"] for kw in keywords],
                "keyword_details": keywords
            })

        return pd.DataFrame(results)


# ============================================================
# 第三部分：机器学习分类模型
# ============================================================

class StateCapacityClassifier:
    """国家能力分类器"""

    def __init__(self):
        self.labeler = RuleBasedLabeler()
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.models = {
            "naive_bayes": MultinomialNB(),
            "svm": LinearSVC(max_iter=10000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.label_encoder = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
        self.label_decoder = {0: "汲取能力", 1: "协调能力", 2: "合规能力"}

    def prepare_training_data(self, df, text_column="合同名称", min_confidence=0.6):
        """
        准备训练数据
        使用规则标注的高置信度样本作为训练集
        """
        print("=" * 60)
        print("步骤1: 使用关键词规则自动标注训练集")
        print("=" * 60)

        # 使用规则标注
        labeled_df = self.labeler.label_dataset(df, text_column)

        # 筛选高置信度样本
        high_conf = labeled_df[
            (labeled_df["label"].notna()) &
            (labeled_df["confidence"] >= min_confidence)
        ].copy()

        print(f"\n总样本数: {len(df)}")
        print(f"成功标注样本数: {len(labeled_df[labeled_df['label'].notna()])}")
        print(f"高置信度样本数 (置信度>={min_confidence}): {len(high_conf)}")

        # 统计各类别数量
        print("\n各类别样本分布:")
        label_counts = high_conf["label"].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(high_conf)*100:.1f}%)")

        # 显示每个类别的样本示例
        print("\n各类别样本示例:")
        for label in ["汲取能力", "协调能力", "合规能力"]:
            examples = high_conf[high_conf["label"] == label].head(3)
            print(f"\n【{label}】")
            for _, row in examples.iterrows():
                print(f"  - {row['text'][:50]}...")
                print(f"    关键词: {row['matched_keywords']}")

        return high_conf

    def train(self, training_df, test_size=0.2):
        """训练模型"""
        print("\n" + "=" * 60)
        print("步骤2: 训练机器学习模型")
        print("=" * 60)

        # 准备数据
        X = training_df["text"].fillna("")
        y = training_df["label"].map(self.label_encoder)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\n训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        # TF-IDF特征提取
        print("\n正在进行TF-IDF特征提取...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"特征维度: {X_train_tfidf.shape[1]}")

        # 训练多个模型并比较
        best_score = 0
        results = {}

        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            model.fit(X_train_tfidf, y_train)
            score = model.score(X_test_tfidf, y_test)
            results[name] = score
            print(f"  准确率: {score:.4f}")

            if score > best_score:
                best_score = score
                self.best_model = model
                best_model_name = name

        print(f"\n最佳模型: {best_model_name} (准确率: {best_score:.4f})")

        # 详细评估最佳模型
        print("\n" + "=" * 60)
        print("步骤3: 模型评估")
        print("=" * 60)

        y_pred = self.best_model.predict(X_test_tfidf)

        print("\n分类报告:")
        target_names = ["汲取能力", "协调能力", "合规能力"]
        print(classification_report(y_test, y_pred, target_names=target_names))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm, index=target_names, columns=target_names))

        return results

    def predict(self, texts):
        """预测新样本"""
        if isinstance(texts, str):
            texts = [texts]

        texts = pd.Series(texts).fillna("")
        X_tfidf = self.vectorizer.transform(texts)

        predictions = self.best_model.predict(X_tfidf)

        # 获取预测概率（如果模型支持）
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(X_tfidf)
        else:
            # SVM使用decision_function
            decision = self.best_model.decision_function(X_tfidf)
            # 简单的softmax转换
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probabilities = exp_decision / exp_decision.sum(axis=1, keepdims=True)

        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            # 也用规则方法找关键词
            _, _, keywords = self.labeler.label_contract(text)

            results.append({
                "text": text,
                "predicted_label": self.label_decoder[pred],
                "confidence": float(max(probs)),
                "probabilities": {
                    "汲取能力": float(probs[0]),
                    "协调能力": float(probs[1]),
                    "合规能力": float(probs[2])
                },
                "matched_keywords": [kw["keyword"] for kw in keywords]
            })

        return results

    def classify_full_dataset(self, df, text_column="合同名称"):
        """对完整数据集进行分类"""
        print("\n" + "=" * 60)
        print("步骤4: 对全部合同进行分类预测")
        print("=" * 60)

        texts = df[text_column].fillna("")
        predictions = self.predict(texts)

        # 创建结果DataFrame
        result_df = df.copy()
        result_df["预测类别"] = [p["predicted_label"] for p in predictions]
        result_df["预测置信度"] = [p["confidence"] for p in predictions]
        result_df["匹配关键词"] = [",".join(p["matched_keywords"]) for p in predictions]
        result_df["汲取能力概率"] = [p["probabilities"]["汲取能力"] for p in predictions]
        result_df["协调能力概率"] = [p["probabilities"]["协调能力"] for p in predictions]
        result_df["合规能力概率"] = [p["probabilities"]["合规能力"] for p in predictions]

        # 统计结果
        print("\n分类结果统计:")
        category_counts = result_df["预测类别"].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} ({count/len(result_df)*100:.1f}%)")

        return result_df


# ============================================================
# 第四部分：主程序
# ============================================================

def main():
    """主函数"""
    print("=" * 70)
    print("国家能力合同分类系统")
    print("基于 Berwick & Christia (2018) 'State Capacity Redux'")
    print("=" * 70)

    # 打印关键词来源说明
    print_keyword_sources()

    # 加载数据
    print("\n\n" + "=" * 70)
    print("加载合同数据")
    print("=" * 70)

    df = pd.read_excel("/home/user/MOR-SI/2015.xls")
    print(f"加载数据: {len(df)} 条合同记录")

    # 初始化分类器
    classifier = StateCapacityClassifier()

    # 准备训练数据
    training_df = classifier.prepare_training_data(df, min_confidence=0.6)

    # 检查是否有足够的训练数据
    min_samples_per_class = training_df["label"].value_counts().min()
    if min_samples_per_class < 10:
        print(f"\n警告: 某些类别样本过少 (最少: {min_samples_per_class})")
        print("降低置信度阈值以获取更多训练样本...")
        training_df = classifier.prepare_training_data(df, min_confidence=0.5)

    # 训练模型
    if len(training_df) >= 50:
        classifier.train(training_df)

        # 对全部数据进行分类
        result_df = classifier.classify_full_dataset(df)

        # 保存结果
        output_path = "/home/user/MOR-SI/合同分类结果.xlsx"
        result_df.to_excel(output_path, index=False)
        print(f"\n结果已保存至: {output_path}")

        # 显示部分预测结果示例
        print("\n" + "=" * 60)
        print("预测结果示例")
        print("=" * 60)

        for capacity in ["汲取能力", "协调能力", "合规能力"]:
            print(f"\n【{capacity}】 (前5个高置信度样本):")
            samples = result_df[result_df["预测类别"] == capacity].nlargest(5, "预测置信度")
            for _, row in samples.iterrows():
                print(f"  - {row['合同名称'][:60]}...")
                print(f"    置信度: {row['预测置信度']:.3f}, 关键词: {row['匹配关键词'][:50]}")
    else:
        print(f"\n错误: 训练样本不足 ({len(training_df)}), 需要至少50个样本")

    return classifier, result_df if 'result_df' in locals() else None


if __name__ == "__main__":
    classifier, results = main()
