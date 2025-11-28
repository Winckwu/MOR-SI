#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
如何使用训练好的模型对新数据进行分类
=================================

本脚本展示如何将训练好的模型应用到你的其他数据上
"""

import pandas as pd
import pickle


def simple_tokenizer(text):
    """分词器（必须与训练时相同）"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    return chars + bigrams


def load_model(model_path='claude_llm_ml_model.pkl'):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ 模型加载成功: {model_path}")
    return data


def predict_contracts(texts, model_data):
    """
    对合同列表进行分类预测

    参数:
        texts: 合同名称列表
        model_data: 加载的模型数据

    返回:
        预测结果列表
    """
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    label_map_reverse = model_data['label_map_reverse']

    # 向量化
    X = vectorizer.transform(texts)

    # 预测
    predictions = model.predict(X)

    # 转换为标签
    results = []
    for text, pred in zip(texts, predictions):
        results.append({
            'text': text,
            'predicted_label': label_map_reverse[pred]
        })

    return results


def predict_excel(input_file, output_file, text_column='合同名称', model_path='claude_llm_ml_model.pkl'):
    """
    对Excel文件进行批量分类

    参数:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径
        text_column: 合同名称所在的列名
        model_path: 模型文件路径
    """
    print(f"\n{'='*60}")
    print("政府采购合同国家能力分类器")
    print(f"{'='*60}")

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    model_data = load_model(model_path)

    # 2. 读取数据
    print(f"\n[2/4] 读取数据: {input_file}")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("支持的文件格式: .xlsx, .xls, .csv")

    print(f"   数据量: {len(df)} 条")
    print(f"   文本列: {text_column}")

    # 3. 预测
    print(f"\n[3/4] 进行分类预测...")
    texts = df[text_column].astype(str).tolist()
    results = predict_contracts(texts, model_data)

    # 添加预测结果
    df['国家能力分类'] = [r['predicted_label'] for r in results]

    # 统计
    print(f"\n   分类结果统计:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = (df['国家能力分类'] == label).sum()
        print(f"   - {label}: {count} ({count/len(df)*100:.1f}%)")

    # 4. 保存结果
    print(f"\n[4/4] 保存结果: {output_file}")
    df.to_excel(output_file, index=False)

    print(f"\n{'='*60}")
    print("✓ 分类完成!")
    print(f"{'='*60}")

    return df


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":

    # 方式1: 对单条合同分类
    print("\n" + "="*60)
    print("示例1: 对单条合同分类")
    print("="*60)

    model_data = load_model('claude_llm_ml_model.pkl')

    test_contracts = [
        "南京市国税局金税三期广域网线路升级采购合同",
        "合肥市轨道交通2号线龙岗停车场工程施工监理",
        "淮北师范大学2015年滨湖校区图书馆建设设备采购",
        "办公设备采购合同",
        "某某县人民医院医疗设备采购项目"
    ]

    results = predict_contracts(test_contracts, model_data)

    print("\n分类结果:")
    for r in results:
        print(f"  • {r['text'][:40]}...")
        print(f"    → {r['predicted_label']}")


    # 方式2: 对Excel文件批量分类
    print("\n\n" + "="*60)
    print("示例2: 对Excel文件批量分类")
    print("="*60)

    # 取消下面的注释来处理你的数据文件
    # predict_excel(
    #     input_file='你的数据.xlsx',       # 输入文件
    #     output_file='分类结果.xlsx',       # 输出文件
    #     text_column='合同名称',            # 合同名称所在的列名
    #     model_path='claude_llm_ml_model.pkl'  # 模型文件
    # )

    print("\n如需处理你的数据，请修改上面的参数并运行")


# ============================================================
# 命令行使用
# ============================================================
"""
命令行使用方式:

python 使用说明_预测新数据.py

或者在Python中直接调用:

>>> from 使用说明_预测新数据 import predict_excel
>>> predict_excel('你的数据.xlsx', '分类结果.xlsx', '合同名称')
"""
