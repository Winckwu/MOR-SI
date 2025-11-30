#!/usr/bin/env python3
"""
政府采购合同国家能力分类器 - 简化版
直接使用预训练模型进行分类

使用方法:
    python classify_with_model.py 2016.csv
    python classify_with_model.py 2017.dta
    python classify_with_model.py 2018.xlsx
"""

import pandas as pd
import pickle
import os
import sys

def load_model(model_path='state_capacity_model.pkl'):
    """加载预训练模型"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['vectorizer'], model_data['model'], model_data['label_encoder']

def load_data(file_path):
    """加载数据文件"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.dta':
        return pd.read_stata(file_path)
    elif ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的格式: {ext}")

def find_columns(df):
    """自动识别列名"""
    buyer_col = contract_col = None
    buyer_names = ['采购人', '采购单位', '采购机构', '购买人']
    contract_names = ['合同名称', '项目名称', '采购项目', '合同标题']

    for col in df.columns:
        col_str = str(col)
        if buyer_col is None:
            for name in buyer_names:
                if name in col_str:
                    buyer_col = col
                    break
        if contract_col is None:
            for name in contract_names:
                if name in col_str:
                    contract_col = col
                    break
    return buyer_col, contract_col

def classify(file_path, output_path=None):
    """分类主函数"""
    print(f"\n{'='*60}")
    print(f"加载模型...")
    vectorizer, model, le = load_model()

    print(f"加载数据: {file_path}")
    df = load_data(file_path)
    print(f"数据量: {len(df)} 条")

    buyer_col, contract_col = find_columns(df)
    if not buyer_col or not contract_col:
        print(f"列名: {list(df.columns)}")
        raise ValueError("无法识别采购人或合同名称列")

    print(f"使用列: 采购人='{buyer_col}', 合同名称='{contract_col}'")

    # 构建文本并预测
    text = df[buyer_col].fillna('').astype(str) + ' ' + df[contract_col].fillna('').astype(str)
    X = vectorizer.transform(text)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # 添加结果
    df['分类预测'] = le.inverse_transform(y_pred)
    df['预测置信度'] = y_proba.max(axis=1)
    for i, cls in enumerate(le.classes_):
        df[f'{cls}_概率'] = y_proba[:, i]

    # 统计
    print(f"\n{'-'*40}")
    print("分类结果统计:")
    print(f"{'-'*40}")
    for cls, count in df['分类预测'].value_counts().items():
        print(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

    # 保存
    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '_classified.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")

    return df

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python classify_with_model.py <数据文件>")
        print("示例: python classify_with_model.py 2016.csv")
        sys.exit(1)

    classify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
