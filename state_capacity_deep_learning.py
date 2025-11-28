#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国家能力合同分类器 - 深度学习版本
基于 Berwick & Christia (2018) "State Capacity Redux" 论文框架

使用 TextCNN 和 LSTM 进行文本分类
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 关键词词典（与机器学习版本相同）
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


# ============================================================
# 词汇表和数据处理
# ============================================================

class Vocabulary:
    """词汇表类"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        self.char_count = Counter()

    def build_vocab(self, texts, min_freq=1):
        """从文本构建词汇表"""
        for text in texts:
            if pd.isna(text):
                continue
            for char in str(text):
                self.char_count[char] += 1

        idx = len(self.char2idx)
        for char, count in self.char_count.items():
            if count >= min_freq and char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        print(f"词汇表大小: {len(self.char2idx)}")

    def encode(self, text, max_len=100):
        """将文本编码为索引序列"""
        if pd.isna(text):
            return [0] * max_len

        text = str(text)
        encoded = []
        for char in text[:max_len]:
            encoded.append(self.char2idx.get(char, 1))  # 1 = <UNK>

        # 填充或截断
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))

        return encoded

    def __len__(self):
        return len(self.char2idx)


class ContractDataset(Dataset):
    """合同数据集"""

    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]

        encoded = self.vocab.encode(text, self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============================================================
# 深度学习模型
# ============================================================

class TextCNN(nn.Module):
    """TextCNN模型 - 用于文本分类"""

    def __init__(self, vocab_size, embed_dim=128, num_classes=3,
                 filter_sizes=[2, 3, 4, 5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)

        # 多个卷积核
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, seq_len - fs + 1)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 拼接所有卷积核的输出
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class LSTMClassifier(nn.Module):
    """LSTM分类模型"""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的隐藏状态
        if self.lstm.bidirectional:
            # 拼接前向和后向的最后隐藏状态
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        x = self.dropout(hidden)
        x = self.fc(x)

        return x


# ============================================================
# 规则标注器
# ============================================================

class RuleBasedLabeler:
    """基于规则的标注器"""

    def __init__(self):
        self.keyword_dict = get_all_keywords()
        self.all_keywords = sorted(self.keyword_dict.keys(), key=len, reverse=True)

    def label_contract(self, text):
        """标注单个合同"""
        if pd.isna(text):
            return None, 0

        text = str(text)
        capacity_scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}

        for keyword in self.all_keywords:
            if keyword in text:
                capacity = self.keyword_dict[keyword]
                capacity_scores[capacity] += len(keyword)

        total = sum(capacity_scores.values())
        if total == 0:
            return None, 0

        max_capacity = max(capacity_scores, key=capacity_scores.get)
        confidence = capacity_scores[max_capacity] / total

        if confidence >= 0.5:
            return max_capacity, confidence
        return None, confidence


# ============================================================
# 训练和评估
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """训练模型"""
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    return history


def evaluate_model(model, test_loader, label_names):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    print("\n混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(pd.DataFrame(cm, index=label_names, columns=label_names))

    return all_preds, all_labels


def predict_all(model, vocab, df, text_column="合同名称", max_len=100, batch_size=128):
    """预测所有数据"""
    model.eval()
    texts = df[text_column].fillna("")

    # 编码所有文本
    encoded = [vocab.encode(text, max_len) for text in texts]
    tensor_data = torch.tensor(encoded, dtype=torch.long)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("国家能力合同分类系统 - 深度学习版本")
    print("基于 Berwick & Christia (2018) 'State Capacity Redux'")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_excel("/home/user/MOR-SI/2015.xls")
    print(f"总数据量: {len(df)}")

    # 使用规则标注训练数据
    print("\n使用规则标注训练数据...")
    labeler = RuleBasedLabeler()
    label_map = {"汲取能力": 0, "协调能力": 1, "合规能力": 2}
    label_names = ["汲取能力", "协调能力", "合规能力"]

    labeled_data = []
    for _, row in df.iterrows():
        text = row["合同名称"]
        label, conf = labeler.label_contract(text)
        if label and conf >= 0.6:
            labeled_data.append({"text": text, "label": label_map[label]})

    labeled_df = pd.DataFrame(labeled_data)
    print(f"标注样本数: {len(labeled_df)}")
    print(f"类别分布: {labeled_df['label'].value_counts().to_dict()}")

    # 构建词汇表
    print("\n构建词汇表...")
    vocab = Vocabulary()
    vocab.build_vocab(df["合同名称"])

    # 划分数据集
    train_df, test_df = train_test_split(labeled_df, test_size=0.2, random_state=42,
                                         stratify=labeled_df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42,
                                        stratify=train_df["label"])

    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 创建数据加载器
    max_len = 100
    batch_size = 64

    train_dataset = ContractDataset(train_df["text"], train_df["label"], vocab, max_len)
    val_dataset = ContractDataset(val_df["text"], val_df["label"], vocab, max_len)
    test_dataset = ContractDataset(test_df["text"], test_df["label"], vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 选择模型
    print("\n" + "=" * 60)
    print("训练 TextCNN 模型")
    print("=" * 60)

    model_cnn = TextCNN(
        vocab_size=len(vocab),
        embed_dim=128,
        num_classes=3,
        filter_sizes=[2, 3, 4, 5],
        num_filters=100,
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

    print(f"\n模型参数量: {sum(p.numel() for p in model_cnn.parameters()):,}")

    # 训练TextCNN
    history_cnn = train_model(model_cnn, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # 评估TextCNN
    print("\n" + "=" * 60)
    print("TextCNN 评估结果")
    print("=" * 60)
    model_cnn.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model_cnn, test_loader, label_names)

    # 训练LSTM
    print("\n" + "=" * 60)
    print("训练 BiLSTM 模型")
    print("=" * 60)

    model_lstm = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True
    ).to(device)

    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

    print(f"\n模型参数量: {sum(p.numel() for p in model_lstm.parameters()):,}")

    history_lstm = train_model(model_lstm, train_loader, val_loader, criterion, optimizer_lstm, num_epochs=10)

    # 评估LSTM
    print("\n" + "=" * 60)
    print("BiLSTM 评估结果")
    print("=" * 60)
    model_lstm.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model_lstm, test_loader, label_names)

    # 对全部数据预测
    print("\n" + "=" * 60)
    print("对全部数据进行预测")
    print("=" * 60)

    predictions, probabilities = predict_all(model_lstm, vocab, df)

    result_df = df.copy()
    result_df["DL_预测类别"] = [label_names[p] for p in predictions]
    result_df["DL_预测置信度"] = [max(p) for p in probabilities]
    result_df["DL_汲取能力概率"] = [p[0] for p in probabilities]
    result_df["DL_协调能力概率"] = [p[1] for p in probabilities]
    result_df["DL_合规能力概率"] = [p[2] for p in probabilities]

    print("\n深度学习分类结果统计:")
    print(result_df["DL_预测类别"].value_counts())

    # 保存结果
    output_path = "/home/user/MOR-SI/合同分类结果_深度学习.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")

    return model_cnn, model_lstm, vocab, result_df


if __name__ == "__main__":
    try:
        models = main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
