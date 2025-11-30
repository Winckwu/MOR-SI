#!/usr/bin/env python3
"""
政府采购合同国家能力分类器
用于将训练好的模型应用到新年份的数据集

使用方法:
    python classify_new_data.py 2016.dta
    python classify_new_data.py 2017.csv
    python classify_new_data.py data/2018.dta --output results/2018_classified.csv

支持格式: .dta (Stata), .csv, .xlsx (Excel)
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class StateCapacityClassifier:
    """国家能力分类器"""

    def __init__(self, model_path=None):
        self.vectorizer = None
        self.model = None
        self.le = None
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_training_data(self, batch_pattern='classification_batch*.csv'):
        """加载标注好的训练数据"""
        print("=" * 60)
        print("加载训练数据...")
        print("=" * 60)

        all_files = glob.glob(batch_pattern)
        if not all_files:
            raise FileNotFoundError(f"未找到训练数据文件: {batch_pattern}")

        dfs = []
        for file in sorted(all_files):
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                print(f"  警告: 读取 {file} 失败: {e}")

        train_df = pd.concat(dfs, ignore_index=True)
        train_df = train_df.drop_duplicates(subset=['序号'])
        train_df = train_df[train_df['分类'] != '其他']
        train_df['text'] = train_df['采购人'].fillna('') + ' ' + train_df['合同名称'].fillna('')

        print(f"加载完成: {len(train_df)} 条训练数据")
        print(f"\n类别分布:")
        for cls, count in train_df['分类'].value_counts().items():
            print(f"  {cls}: {count} ({count/len(train_df)*100:.2f}%)")

        return train_df

    def train(self, train_df=None, batch_pattern='classification_batch*.csv'):
        """训练模型"""
        if train_df is None:
            train_df = self.load_training_data(batch_pattern)

        print("\n" + "=" * 60)
        print("训练模型...")
        print("=" * 60)

        # 标签编码
        self.le = LabelEncoder()
        y = self.le.fit_transform(train_df['分类'])

        # TF-IDF向量化 (字符级 2-4 grams，最优配置)
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=15000,
            min_df=2
        )
        X = self.vectorizer.fit_transform(train_df['text'])

        # LogisticRegression (与Claude人工审核一致率最高的模型)
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        self.model.fit(X, y)

        self.is_trained = True
        print(f"模型训练完成!")
        print(f"类别: {list(self.le.classes_)}")

    def save_model(self, path='state_capacity_model.pkl'):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")

        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'label_encoder': self.le
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {path}")

    def load_model(self, path='state_capacity_model.pkl'):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.le = model_data['label_encoder']
        self.is_trained = True
        print(f"模型已从 {path} 加载")

    def load_new_data(self, file_path):
        """加载新数据文件"""
        print(f"\n加载数据文件: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.dta':
            df = pd.read_stata(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        print(f"数据量: {len(df)} 条")
        print(f"列名: {list(df.columns)[:10]}...")

        return df

    def find_columns(self, df):
        """自动识别采购人和合同名称列"""
        buyer_col = None
        contract_col = None

        # 常见列名
        buyer_names = ['采购人', '采购单位', '采购机构', '购买人', 'buyer', 'purchaser']
        contract_names = ['合同名称', '项目名称', '采购项目', '合同标题', 'contract', 'title', 'project']

        for col in df.columns:
            col_lower = str(col).lower()
            if buyer_col is None:
                for name in buyer_names:
                    if name in col_lower or col_lower in name:
                        buyer_col = col
                        break
            if contract_col is None:
                for name in contract_names:
                    if name in col_lower or col_lower in name:
                        contract_col = col
                        break

        return buyer_col, contract_col

    def classify(self, df, buyer_col=None, contract_col=None):
        """对新数据进行分类"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 或 load_model() 方法")

        # 自动识别列名
        if buyer_col is None or contract_col is None:
            auto_buyer, auto_contract = self.find_columns(df)
            buyer_col = buyer_col or auto_buyer
            contract_col = contract_col or auto_contract

        if buyer_col is None:
            raise ValueError("无法识别采购人列，请手动指定 buyer_col 参数")
        if contract_col is None:
            raise ValueError("无法识别合同名称列，请手动指定 contract_col 参数")

        print(f"\n使用列: 采购人='{buyer_col}', 合同名称='{contract_col}'")

        # 构建文本特征
        df['_text'] = df[buyer_col].fillna('').astype(str) + ' ' + df[contract_col].fillna('').astype(str)

        # 向量化和预测
        X = self.vectorizer.transform(df['_text'])
        y_pred = self.model.predict(X)

        # 获取预测概率
        y_proba = self.model.predict_proba(X)

        # 添加预测结果
        df['分类预测'] = self.le.inverse_transform(y_pred)
        df['预测置信度'] = y_proba.max(axis=1)

        # 添加各类别概率
        for i, cls in enumerate(self.le.classes_):
            df[f'{cls}_概率'] = y_proba[:, i]

        # 清理临时列
        df = df.drop(columns=['_text'])

        return df

    def classify_file(self, input_file, output_file=None, buyer_col=None, contract_col=None):
        """分类文件并保存结果"""
        print("\n" + "=" * 60)
        print("开始分类...")
        print("=" * 60)

        # 加载数据
        df = self.load_new_data(input_file)

        # 分类
        df = self.classify(df, buyer_col, contract_col)

        # 统计结果
        print("\n" + "-" * 40)
        print("分类结果统计:")
        print("-" * 40)
        for cls, count in df['分类预测'].value_counts().items():
            print(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

        # 保存结果
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_classified.csv"

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")

        return df


def main():
    parser = argparse.ArgumentParser(
        description='政府采购合同国家能力分类器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分类单个文件
  python classify_new_data.py 2016.dta

  # 指定输出文件
  python classify_new_data.py 2017.csv --output 2017_result.csv

  # 手动指定列名
  python classify_new_data.py data.xlsx --buyer 采购单位 --contract 项目名称

  # 先训练再保存模型
  python classify_new_data.py --train --save-model my_model.pkl

  # 使用已保存的模型
  python classify_new_data.py 2018.dta --model my_model.pkl
        """
    )

    parser.add_argument('input_file', nargs='?', help='要分类的数据文件 (.dta/.csv/.xlsx)')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--buyer', help='采购人列名')
    parser.add_argument('--contract', help='合同名称列名')
    parser.add_argument('--model', '-m', help='已保存的模型文件路径')
    parser.add_argument('--save-model', help='保存模型到指定路径')
    parser.add_argument('--train', action='store_true', help='重新训练模型')
    parser.add_argument('--batch-pattern', default='classification_batch*.csv',
                        help='训练数据文件模式 (默认: classification_batch*.csv)')

    args = parser.parse_args()

    # 创建分类器
    classifier = StateCapacityClassifier()

    # 加载或训练模型
    if args.model and os.path.exists(args.model):
        classifier.load_model(args.model)
    else:
        classifier.train(batch_pattern=args.batch_pattern)

    # 保存模型
    if args.save_model:
        classifier.save_model(args.save_model)

    # 分类文件
    if args.input_file:
        classifier.classify_file(
            args.input_file,
            output_file=args.output,
            buyer_col=args.buyer,
            contract_col=args.contract
        )
    elif not args.train and not args.save_model:
        parser.print_help()


if __name__ == '__main__':
    main()
