#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - 预测脚本
使用训练好的模型对新数据进行分类

使用方法:
    python predict.py --data 你的数据.xlsx --column 合同名称

输出:
    你的数据_分类结果.xlsx
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os


def simple_tokenizer(text):
    """简单的中文分词器（字符级 + bigram）"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


class ContractClassifier:
    """合同分类器"""

    def __init__(self, model_path=None):
        """
        初始化分类器

        参数:
            model_path: 模型文件路径，默认使用方案2模型（效果更好）
        """
        if model_path is None:
            # 默认使用方案2模型（效果更好）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "model_llm_semantic.pkl")

        self.load_model(model_path)

    def load_model(self, model_path):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.label_map = model_data['label_map']
        self.label_map_reverse = model_data['label_map_reverse']
        self.model_info = {
            "model_name": model_data.get('model_name', 'Unknown'),
            "accuracy": model_data.get('accuracy', 0),
            "f1": model_data.get('f1', 0),
            "approach": model_data.get('approach', 'Unknown'),
            "created_at": model_data.get('created_at', 'Unknown')
        }

        print(f"模型已加载: {model_path}")
        print(f"  方案: {self.model_info['approach']}")
        print(f"  模型类型: {self.model_info['model_name']}")
        print(f"  准确率: {self.model_info['accuracy']:.4f}")
        print(f"  F1分数: {self.model_info['f1']:.4f}")

    def predict(self, texts):
        """
        预测合同类别

        参数:
            texts: 文本列表或单个文本

        返回:
            预测结果列表
        """
        if isinstance(texts, str):
            texts = [texts]

        # 转换为字符串
        texts = [str(t) if t else "" for t in texts]

        # TF-IDF转换
        X_tfidf = self.vectorizer.transform(texts)

        # 预测
        predictions = self.model.predict(X_tfidf)

        # 如果模型支持概率输出
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_tfidf)
        else:
            # SVM使用decision_function
            decision = self.model.decision_function(X_tfidf)
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probabilities = exp_decision / exp_decision.sum(axis=1, keepdims=True)

        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            result = {
                "text": text,
                "predicted_label": self.label_map_reverse[pred],
                "confidence": float(max(probs)),
                "probabilities": {
                    "汲取能力": float(probs[0]),
                    "协调能力": float(probs[1]),
                    "合规能力": float(probs[2])
                }
            }
            results.append(result)

        return results

    def predict_dataframe(self, df, text_column="合同名称"):
        """
        对DataFrame进行预测

        参数:
            df: pandas DataFrame
            text_column: 包含合同名称的列名

        返回:
            添加了预测结果的DataFrame
        """
        texts = df[text_column].tolist()
        results = self.predict(texts)

        result_df = df.copy()
        result_df["预测类别"] = [r["predicted_label"] for r in results]
        result_df["预测置信度"] = [r["confidence"] for r in results]
        result_df["汲取能力概率"] = [r["probabilities"]["汲取能力"] for r in results]
        result_df["协调能力概率"] = [r["probabilities"]["协调能力"] for r in results]
        result_df["合规能力概率"] = [r["probabilities"]["合规能力"] for r in results]

        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="国家能力合同分类器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型（方案2，效果更好）
  python predict.py --data 你的数据.xlsx --column 合同名称

  # 使用方案1模型
  python predict.py --data 你的数据.xlsx --column 合同名称 --model model_rule_based.pkl

  # 指定输出文件
  python predict.py --data 你的数据.xlsx --column 合同名称 --output 结果.xlsx
        """
    )
    parser.add_argument("--data", type=str, required=True, help="数据文件路径 (xlsx/xls)")
    parser.add_argument("--column", type=str, default="合同名称", help="合同名称列名")
    parser.add_argument("--model", type=str, default=None, help="模型文件路径（默认使用方案2模型）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")

    args = parser.parse_args()

    # 加载模型
    print("=" * 60)
    print("国家能力合同分类器")
    print("=" * 60)

    classifier = ContractClassifier(args.model)

    # 加载数据
    print(f"\n加载数据: {args.data}")
    if args.data.endswith('.xlsx'):
        df = pd.read_excel(args.data)
    else:
        df = pd.read_excel(args.data)
    print(f"数据量: {len(df)} 条")

    # 预测
    print(f"\n开始预测...")
    result_df = classifier.predict_dataframe(df, args.column)

    # 统计结果
    print("\n预测结果分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = (result_df["预测类别"] == label).sum()
        print(f"  {label}: {count} ({count/len(result_df)*100:.1f}%)")

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.data)[0]
        output_path = f"{base_name}_分类结果.xlsx"

    result_df.to_excel(output_path, index=False)
    print(f"\n结果已保存: {output_path}")

    return result_df


if __name__ == "__main__":
    main()
