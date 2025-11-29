#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Claude API进行真实语义标注
解决自我循环问题：用LLM真正理解合同内容，而非规则匹配
"""

import pandas as pd
import json
import os
import time
from datetime import datetime

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("警告: 未安装anthropic库，请运行: pip install anthropic")


CLASSIFICATION_PROMPT = """你是一个政府采购合同分类专家。请根据Berwick & Christia (2018) "State Capacity Redux"论文的国家能力框架，将以下合同分类为三类之一：

## 分类标准

1. **汲取能力 (Extractive Capacity)**
   - 定义：国家从社会获取资源的能力
   - 典型特征：税收、财政、海关、统计普查、土地调查确权、资产评估、审计
   - 关键机构：税务局、财政局、海关、统计局、审计局、国土局

2. **协调能力 (Coordination Capacity)**
   - 定义：国家组织集体行动、提供公共品的能力
   - 典型特征：基础设施建设（道路、水利、电力、通信）、政府办公、信息化建设
   - 关键机构：交通局、建设局、水利局、电力局、发改委、机关事务局

3. **合规能力 (Compliance Capacity)**
   - 定义：确保公民和官僚服从国家目标的能力
   - 典型特征：教育、医疗卫生、公安执法、监督检查、社会保障、农林牧渔管理
   - 关键机构：学校、医院、公安局、法院、教育局、卫健委、民政局

## 分类要点

- 根据**采购目的和受益对象**判断，而非单纯看采购物品
- 同样是买电脑：税务局买→汲取能力；学校买→合规能力；政府办公室买→协调能力
- 基础设施建设（道路、桥梁、水电）→协调能力
- 教育医疗设备、执法装备→合规能力
- 税务系统、财政软件、统计调查→汲取能力

## 待分类合同

"""


def create_batch_prompt(contracts):
    """创建批量分类提示"""
    prompt = CLASSIFICATION_PROMPT
    prompt += "请对以下合同逐一分类，直接返回JSON格式：\n\n"

    for i, contract in enumerate(contracts):
        prompt += f"{i+1}. {contract}\n"

    prompt += """
返回格式（严格JSON）：
```json
[
  {"id": 1, "label": "汲取能力/协调能力/合规能力", "reason": "简短理由"},
  ...
]
```

只返回JSON，不要其他内容。"""
    return prompt


def classify_batch_with_claude(client, contracts, batch_id, model="claude-sonnet-4-20250514"):
    """使用Claude API对一批合同进行分类"""
    prompt = create_batch_prompt(contracts)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text

        # 提取JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        results = json.loads(json_str.strip())
        return results

    except Exception as e:
        print(f"  批次 {batch_id} 出错: {e}")
        return None


def run_claude_labeling(data_path, output_path, num_samples=10000, batch_size=50):
    """
    运行Claude标注流程

    Args:
        data_path: 数据文件路径
        output_path: 输出文件路径
        num_samples: 标注样本数
        batch_size: 每批处理数量
    """
    if not HAS_ANTHROPIC:
        print("错误: 请先安装anthropic库")
        print("运行: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置ANTHROPIC_API_KEY环境变量")
        print("运行: export ANTHROPIC_API_KEY='your-api-key'")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # 加载数据
    print("=" * 70)
    print("Claude真实语义标注")
    print("=" * 70)
    print(f"\n加载数据: {data_path}")

    df = pd.read_excel(data_path)
    print(f"总记录数: {len(df)}")

    # 抽取样本
    if num_samples < len(df):
        sample_df = df.sample(n=num_samples, random_state=42)
    else:
        sample_df = df

    print(f"标注样本数: {len(sample_df)}")

    contracts = sample_df['合同名称'].tolist()

    # 分批处理
    num_batches = (len(contracts) + batch_size - 1) // batch_size
    print(f"批次数: {num_batches} (每批 {batch_size} 条)")

    all_results = []

    for batch_id in range(num_batches):
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, len(contracts))
        batch_contracts = contracts[start_idx:end_idx]

        print(f"\n处理批次 {batch_id + 1}/{num_batches} ({start_idx+1}-{end_idx})")

        results = classify_batch_with_claude(client, batch_contracts, batch_id + 1)

        if results:
            for r in results:
                all_results.append({
                    "index": start_idx + r["id"] - 1,
                    "contract": batch_contracts[r["id"] - 1],
                    "label": r["label"],
                    "reason": r.get("reason", "")
                })
            print(f"  成功: {len(results)} 条")
        else:
            # 标记失败的
            for i, c in enumerate(batch_contracts):
                all_results.append({
                    "index": start_idx + i,
                    "contract": c,
                    "label": "未分类",
                    "reason": "API调用失败"
                })
            print(f"  失败，已标记为未分类")

        # 避免API限流
        time.sleep(1)

        # 每100批保存一次中间结果
        if (batch_id + 1) % 100 == 0:
            temp_path = output_path.replace(".json", f"_temp_{batch_id+1}.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  已保存中间结果: {temp_path}")

    # 保存最终结果
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "source": data_path,
                "num_samples": len(all_results),
                "labeled_by": "Claude API",
                "model": "claude-sonnet-4-20250514",
                "timestamp": datetime.now().isoformat(),
                "framework": "Berwick & Christia (2018) State Capacity Redux"
            },
            "labels": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"标注结果已保存: {output_path}")

    # 统计
    label_counts = {}
    for r in all_results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n标注分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(all_results)*100:.1f}%)")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用Claude API进行合同分类标注")
    parser.add_argument("--data", type=str, default="/home/user/MOR-SI/2015.xls", help="数据文件路径")
    parser.add_argument("--output", type=str, default="/home/user/MOR-SI/claude_labels_10000.json", help="输出文件路径")
    parser.add_argument("--num", type=int, default=10000, help="标注样本数")
    parser.add_argument("--batch", type=int, default=50, help="每批处理数量")

    args = parser.parse_args()

    run_claude_labeling(args.data, args.output, args.num, args.batch)
