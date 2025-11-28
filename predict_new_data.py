#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - 预测新数据
使用训练好的模型对新数据集进行分类

使用方法:
    python predict_new_data.py 新数据.xlsx
    python predict_new_data.py 新数据.xlsx --column 合同名称 --output 分类结果.xlsx
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import argparse
from collections import Counter

# ============================================================
# 关键词词典（与训练时相同）
# ============================================================

KEYWORD_DICTIONARY = {
    "汲取能力": {
        "keywords": {
            "税收财政类": ["税", "税收", "税务", "财税", "财政", "国税", "地税",
                       "征收", "缴纳", "纳税", "税源", "税基"],
            "预算收入类": ["预算", "决算", "收入", "岁入", "资金", "经费", "拨款"],
            "土地资源类": ["土地调查", "地籍", "确权", "不动产登记", "土地登记",
                       "土地变更", "耕地", "宅基地", "国土"],
            "统计普查类": ["统计", "普查", "调查", "登记", "年鉴", "数据采集"],
            "资产评估类": ["资产", "评估", "估价", "清产核资", "产权"],
            "审计监管类": ["审计", "稽查", "核查", "查账"],
            "关税贸易类": ["关税", "海关", "进出口", "口岸", "边检"]
        }
    },
    "协调能力": {
        "keywords": {
            "交通基础设施": ["公路", "铁路", "高速", "道路", "桥梁", "隧道",
                        "轨道交通", "地铁", "机场", "港口", "码头", "航道"],
            "市政基础设施": ["水利", "供水", "排水", "污水", "给水", "自来水",
                        "电力", "供电", "电网", "变电", "输电",
                        "燃气", "天然气", "供暖", "供热", "管网"],
            "信息基础设施": ["通信", "网络", "信息化", "数字化", "宽带",
                        "光纤", "基站", "覆盖", "信号"],
            "行政办公类": ["行政", "办公", "机关", "政务", "政府"],
            "规划发展类": ["规划", "建设", "工程", "项目", "发展", "改造",
                       "重建", "新建", "扩建", "城镇化", "城市化"],
            "产业园区类": ["产业", "园区", "开发区", "经济区", "工业"],
            "标准检测类": ["标准", "规范", "检测", "检验", "认证", "质量",
                       "计量", "校准", "鉴定"]
        }
    },
    "合规能力": {
        "keywords": {
            "监督监控类": ["监督", "监控", "监察", "巡视", "巡查", "督查",
                       "检查", "抽查", "核查", "视察"],
            "考核培训类": ["考核", "考试", "评估", "评价", "绩效", "奖惩",
                       "培训", "教育培训", "业务培训", "技能培训"],
            "教育服务类": ["教育", "学校", "教学", "课堂", "教室", "师资",
                       "幼儿园", "中学", "小学", "大学", "职业教育",
                       "改薄", "义务教育", "教体"],
            "医疗卫生类": ["医疗", "卫生", "健康", "医院", "诊所", "疾控",
                       "防疫", "疫苗", "疫情", "免疫", "药品", "医药",
                       "计生", "妇幼", "康复"],
            "社会保障类": ["社保", "养老", "低保", "救助", "扶贫", "民政",
                       "残疾", "福利", "殡葬", "救灾", "应急"],
            "执法处罚类": ["执法", "处罚", "惩戒", "强制", "取缔", "查处",
                       "公安", "警察", "警务", "消防", "安防", "监狱"],
            "问责投诉类": ["问责", "责任", "投诉", "举报", "信访", "申诉"]
        }
    }
}


def get_all_keywords():
    """获取所有关键词"""
    keyword_to_capacity = {}
    for capacity_type, content in KEYWORD_DICTIONARY.items():
        for category, keywords in content["keywords"].items():
            for keyword in keywords:
                keyword_to_capacity[keyword] = capacity_type
    return keyword_to_capacity


def find_keywords(text, keyword_dict, all_keywords):
    """在文本中查找匹配的关键词"""
    if pd.isna(text):
        return []
    found = []
    text = str(text)
    for keyword in all_keywords:
        if keyword in text:
            found.append(keyword)
    return found


class StateCapacityPredictor:
    """国家能力分类预测器"""

    def __init__(self, model_path=None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
        """
        self.keyword_dict = get_all_keywords()
        self.all_keywords = sorted(self.keyword_dict.keys(), key=len, reverse=True)
        self.label_names = ["汲取能力", "协调能力", "合规能力"]

        # 模型相关
        self.model = None
        self.vectorizer = None

        # 尝试加载已保存的模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 尝试从默认位置加载
            default_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
            if os.path.exists(default_path):
                self.load_model(default_path)

    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        self.model = saved_data['model']
        self.vectorizer = saved_data['vectorizer']
        print("模型加载成功!")

    def save_model(self, model_path):
        """保存模型"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)
        print(f"模型已保存至: {model_path}")

    def predict_single(self, text):
        """
        预测单个文本

        Returns:
            dict: 包含预测类别、置信度、匹配关键词等
        """
        if pd.isna(text) or not text:
            return {
                "text": text,
                "predicted_label": "协调能力",  # 默认类别
                "confidence": 0.0,
                "matched_keywords": [],
                "method": "default"
            }

        text = str(text)
        matched_keywords = find_keywords(text, self.keyword_dict, self.all_keywords)

        # 如果有模型，使用模型预测
        if self.model and self.vectorizer:
            X = self.vectorizer.transform([text])
            pred = self.model.predict(X)[0]

            # 获取置信度
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                confidence = float(max(probs))
            else:
                # SVM使用decision_function
                decision = self.model.decision_function(X)[0]
                exp_d = np.exp(decision - np.max(decision))
                probs = exp_d / exp_d.sum()
                confidence = float(max(probs))

            return {
                "text": text,
                "predicted_label": self.label_names[pred],
                "confidence": confidence,
                "matched_keywords": matched_keywords,
                "probabilities": {
                    "汲取能力": float(probs[0]),
                    "协调能力": float(probs[1]),
                    "合规能力": float(probs[2])
                },
                "method": "model"
            }
        else:
            # 回退到规则方法
            return self._rule_based_predict(text, matched_keywords)

    def _rule_based_predict(self, text, matched_keywords):
        """基于规则的预测（当没有模型时使用）"""
        capacity_scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}

        for keyword in matched_keywords:
            capacity = self.keyword_dict[keyword]
            capacity_scores[capacity] += len(keyword)

        total = sum(capacity_scores.values())
        if total == 0:
            return {
                "text": text,
                "predicted_label": "协调能力",
                "confidence": 0.33,
                "matched_keywords": matched_keywords,
                "probabilities": {"汲取能力": 0.33, "协调能力": 0.34, "合规能力": 0.33},
                "method": "rule_default"
            }

        max_capacity = max(capacity_scores, key=capacity_scores.get)
        probs = {k: v/total for k, v in capacity_scores.items()}

        return {
            "text": text,
            "predicted_label": max_capacity,
            "confidence": probs[max_capacity],
            "matched_keywords": matched_keywords,
            "probabilities": probs,
            "method": "rule"
        }

    def predict_dataframe(self, df, text_column="合同名称"):
        """
        预测整个数据框

        Args:
            df: pandas DataFrame
            text_column: 包含合同名称的列名

        Returns:
            DataFrame: 添加了预测结果的新数据框
        """
        print(f"\n正在对 {len(df)} 条数据进行分类预测...")

        results = []
        for idx, row in df.iterrows():
            text = row[text_column]
            pred = self.predict_single(text)
            results.append(pred)

            # 进度显示
            if (idx + 1) % 1000 == 0:
                print(f"  已处理: {idx + 1}/{len(df)}")

        # 添加预测结果到原数据框
        result_df = df.copy()
        result_df["预测类别"] = [r["predicted_label"] for r in results]
        result_df["预测置信度"] = [r["confidence"] for r in results]
        result_df["匹配关键词"] = [",".join(r["matched_keywords"]) for r in results]
        result_df["汲取能力概率"] = [r["probabilities"]["汲取能力"] for r in results]
        result_df["协调能力概率"] = [r["probabilities"]["协调能力"] for r in results]
        result_df["合规能力概率"] = [r["probabilities"]["合规能力"] for r in results]

        # 打印统计
        print("\n分类结果统计:")
        counts = result_df["预测类别"].value_counts()
        for cat, count in counts.items():
            print(f"  {cat}: {count} ({count/len(result_df)*100:.1f}%)")

        return result_df


def simple_tokenizer(text):
    """简单的中文分词器（字符级 + bigram）- 必须在模块级别定义以支持pickle"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


def train_and_save_model():
    """训练并保存模型（首次使用时运行）"""
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("=" * 60)
    print("训练模型...")
    print("=" * 60)

    # 加载训练数据
    data_path = os.path.join(os.path.dirname(__file__), "2015.xls")
    if not os.path.exists(data_path):
        print(f"错误: 找不到训练数据 {data_path}")
        return None

    df = pd.read_excel(data_path)
    print(f"加载数据: {len(df)} 条")

    # 使用规则标注
    keyword_dict = get_all_keywords()
    all_keywords = sorted(keyword_dict.keys(), key=len, reverse=True)

    labeled_data = []
    label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}

    for _, row in df.iterrows():
        text = row["合同名称"]
        if pd.isna(text):
            continue

        text = str(text)
        capacity_scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}

        for keyword in all_keywords:
            if keyword in text:
                capacity_scores[keyword_dict[keyword]] += len(keyword)

        total = sum(capacity_scores.values())
        if total > 0:
            max_cap = max(capacity_scores, key=capacity_scores.get)
            conf = capacity_scores[max_cap] / total
            if conf >= 0.6:
                labeled_data.append({"text": text, "label": label_map[max_cap]})

    print(f"高置信度标注样本: {len(labeled_data)} 条")

    # 训练模型
    texts = [d["text"] for d in labeled_data]
    labels = [d["label"] for d in labeled_data]

    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenizer,
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)
    model = LinearSVC(max_iter=10000)
    model.fit(X, labels)

    print("模型训练完成!")

    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer
        }, f)
    print(f"模型已保存至: {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="国家能力合同分类器 - 对新数据进行分类预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python predict_new_data.py 新数据.xlsx
  python predict_new_data.py 新数据.xlsx --column 合同名称
  python predict_new_data.py 新数据.xlsx --output 分类结果.xlsx
  python predict_new_data.py --train  # 首次使用，训练并保存模型
        """
    )

    parser.add_argument("input_file", nargs="?", help="输入的Excel或CSV文件路径")
    parser.add_argument("--column", "-c", default="合同名称", help="合同名称所在的列名 (默认: 合同名称)")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认: 输入文件名_分类结果.xlsx)")
    parser.add_argument("--model", "-m", help="模型文件路径")
    parser.add_argument("--train", action="store_true", help="训练并保存模型")

    args = parser.parse_args()

    # 训练模式
    if args.train:
        train_and_save_model()
        return

    # 预测模式
    if not args.input_file:
        parser.print_help()
        print("\n错误: 请提供输入文件路径")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"错误: 找不到文件 {args.input_file}")
        sys.exit(1)

    # 检查模型是否存在
    model_path = args.model or os.path.join(os.path.dirname(__file__), "trained_model.pkl")
    if not os.path.exists(model_path):
        print("模型文件不存在，正在训练...")
        model_path = train_and_save_model()
        if not model_path:
            print("模型训练失败，将使用规则方法进行分类")

    # 加载数据
    print(f"\n加载数据: {args.input_file}")
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_excel(args.input_file)
    print(f"数据量: {len(df)} 条")

    # 检查列名
    if args.column not in df.columns:
        print(f"错误: 找不到列 '{args.column}'")
        print(f"可用的列: {list(df.columns)}")
        sys.exit(1)

    # 初始化预测器
    predictor = StateCapacityPredictor(model_path)

    # 预测
    result_df = predictor.predict_dataframe(df, text_column=args.column)

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_path = f"{base_name}_分类结果.xlsx"

    result_df.to_excel(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
