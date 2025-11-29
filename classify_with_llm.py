"""
调用 Claude API 对政府采购合同进行国家能力分类

每条合同都由 LLM 理解后给出分类和理由
"""

import pandas as pd
import anthropic
import json
import time
from tqdm import tqdm

# 初始化 Claude 客户端
client = anthropic.Anthropic()

SYSTEM_PROMPT = """你是一个政治学专家，专门研究国家能力(State Capacity)。

根据论文定义，国家能力分为三个维度：
1. **提取能力 (Extractive Capacity)** - 国家获取资源的能力，包括税收征管、资源开采、财政收入、数据/信息采集等
2. **协调能力 (Coordination Capacity)** - 组织集体行动的能力，包括教育、医疗、科研、基础设施建设、公共服务等
3. **合规能力 (Compliance Capacity)** - 确保公众和官员遵从的能力，包括执法、司法、监察、市场监管、环境执法等

你需要根据"采购人"（谁在采购）和"合同内容"（采购什么），判断这笔采购主要增强的是哪种国家能力。

注意：
- 主体的性质很重要：税务局的采购通常增强提取能力，公安局的采购通常增强合规能力
- 采购目的也很重要：同样是买电脑，税务局是为了征税，学校是为了教学
- 综合考虑主体+目的来判断"""

USER_PROMPT_TEMPLATE = """请对以下政府采购合同进行国家能力分类：

采购人：{purchaser}
合同内容：{contract_name}

请用JSON格式回答：
{{"分类": "提取能力/协调能力/合规能力", "理由": "简短理由(20字以内)"}}"""


def classify_single_contract(purchaser, contract_name):
    """调用 Claude API 对单条合同分类"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        purchaser=purchaser,
                        contract_name=contract_name
                    )
                }
            ]
        )

        response_text = message.content[0].text.strip()

        # 解析JSON响应
        # 处理可能的markdown代码块
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        result = json.loads(response_text)
        return result.get('分类', '未知'), result.get('理由', '')

    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试从文本中提取
        if '提取能力' in response_text:
            return '提取能力', response_text[:50]
        elif '合规能力' in response_text:
            return '合规能力', response_text[:50]
        elif '协调能力' in response_text:
            return '协调能力', response_text[:50]
        return '未知', response_text[:50]

    except Exception as e:
        return '错误', str(e)[:50]


def main():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel('2015.xls')
    print(f"数据总量: {len(df)} 条")

    # 可以先测试前几条
    test_mode = input("是否先测试前10条？(y/n): ").strip().lower() == 'y'

    if test_mode:
        df_to_process = df.head(10).copy()
    else:
        df_to_process = df.copy()

    print(f"\n将处理 {len(df_to_process)} 条记录...")
    print("=" * 80)

    # 存储结果
    classifications = []
    reasons = []

    # 逐条处理
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        purchaser = str(row.get('采购人', '未知'))
        contract_name = str(row.get('合同名称', '未知'))

        # 调用 LLM 分类
        classification, reason = classify_single_contract(purchaser, contract_name)

        classifications.append(classification)
        reasons.append(reason)

        # 打印结果（像图片那样）
        print(f"\n采购人: {purchaser[:20]}")
        print(f"合同内容: {contract_name[:40]}")
        print(f"分类: {classification}")
        print(f"理由: {reason}")
        print("-" * 60)

        # 避免API限流
        time.sleep(0.5)

    # 保存结果
    df_to_process['LLM分类'] = classifications
    df_to_process['分类理由'] = reasons

    output_file = '2015_llm_classified.xlsx'
    df_to_process.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 统计
    print("\n=== 分类统计 ===")
    print(pd.Series(classifications).value_counts())


if __name__ == '__main__':
    main()
