"""
用LLM逻辑对政府采购合同进行国家能力三维度分类

分类维度：
1. 提取能力 (Extractive) - 国家获取资源的能力
2. 协调能力 (Coordination) - 组织集体行动的能力
3. 合规能力 (Compliance) - 确保公众和官员遵从的能力

分类逻辑：结合"主体类型"+"采购目的"进行判断
"""

import pandas as pd
import re

# 主体关键词分类
EXTRACTIVE_ACTORS = [
    '税务', '国税', '地税', '海关', '财政', '国土', '资源',
    '矿产', '能源', '石油', '电力', '银行', '金融', '证监',
    '银监', '保监', '外汇', '统计', '审计'
]

COMPLIANCE_ACTORS = [
    '公安', '警察', '法院', '检察', '司法', '监狱', '戒毒',
    '纪委', '监察', '市场监管', '工商', '质监', '药监', '食药',
    '环保', '生态环境', '安监', '应急', '消防', '城管', '交警',
    '边防', '海警', '武警'
]

COORDINATION_ACTORS = [
    '学校', '中学', '小学', '大学', '学院', '教育', '教体',
    '医院', '卫生', '疾控', '疾病预防', '医疗',
    '交通', '公路', '铁路', '民航', '港口',
    '水利', '农业', '农牧', '林业', '渔业',
    '科学院', '研究所', '研究院', '科技',
    '文化', '博物', '图书馆', '档案', '广播', '电视', '新闻',
    '民政', '社保', '人社', '住建', '规划'
]

# 合同内容关键词
EXTRACTIVE_KEYWORDS = [
    '征收', '征管', '税收', '关税', '收费', '罚没', '财务',
    '核算', '审计', '统计', '调查', '普查', '监测', '勘探',
    '测量', '遥感', '数据采集', '信息采集'
]

COMPLIANCE_KEYWORDS = [
    '监控', '监视', '执法', '取证', '审讯', '安防', '安保',
    '警用', '警务', '治安', '刑侦', '侦查', '检测', '鉴定',
    '司法', '法庭', '强制', '处罚', '稽查', '巡查', '巡逻'
]

COORDINATION_KEYWORDS = [
    '教学', '培训', '实训', '实验', '教育', '课程',
    '医疗', '诊断', '治疗', '康复', '护理', '防疫',
    '建设', '工程', '改造', '修缮', '基础设施',
    '科研', '研发', '实验', '试验',
    '服务', '保障', '管理', '办公', '会议', '后勤'
]


def classify_by_actor(purchaser):
    """根据采购主体初步分类"""
    if pd.isna(purchaser):
        return None

    purchaser = str(purchaser)

    # 检查提取能力相关主体
    for keyword in EXTRACTIVE_ACTORS:
        if keyword in purchaser:
            return 'Extractive'

    # 检查合规能力相关主体
    for keyword in COMPLIANCE_ACTORS:
        if keyword in purchaser:
            return 'Compliance'

    # 检查协调能力相关主体
    for keyword in COORDINATION_ACTORS:
        if keyword in purchaser:
            return 'Coordination'

    return None


def classify_by_content(contract_name):
    """根据合同内容辅助分类"""
    if pd.isna(contract_name):
        return None

    contract_name = str(contract_name)

    # 计算各类关键词匹配数
    extractive_score = sum(1 for kw in EXTRACTIVE_KEYWORDS if kw in contract_name)
    compliance_score = sum(1 for kw in COMPLIANCE_KEYWORDS if kw in contract_name)
    coordination_score = sum(1 for kw in COORDINATION_KEYWORDS if kw in contract_name)

    max_score = max(extractive_score, compliance_score, coordination_score)

    if max_score == 0:
        return None

    if extractive_score == max_score:
        return 'Extractive'
    elif compliance_score == max_score:
        return 'Compliance'
    else:
        return 'Coordination'


def classify_state_capacity(row):
    """
    综合分类逻辑（模拟LLM理解）：
    1. 首先看主体是谁 - 主体性质决定了其核心职能
    2. 再看采购目的 - 辅助判断具体用途
    3. 综合判断属于哪种国家能力
    """
    purchaser = row.get('采购人', '')
    contract_name = row.get('合同名称', '')
    industry = row.get('所属行业', '')

    # 第一步：根据主体判断
    actor_class = classify_by_actor(purchaser)

    # 第二步：根据内容判断
    content_class = classify_by_content(contract_name)

    # 第三步：综合判断
    # 规则：主体判断优先，但内容可以覆盖

    # 特殊情况处理：某些采购目的可以覆盖主体判断
    if content_class == 'Compliance' and any(kw in str(contract_name) for kw in ['监控', '执法', '安防', '警用']):
        return 'Compliance'

    if content_class == 'Extractive' and any(kw in str(contract_name) for kw in ['征收', '税收', '监测', '调查', '数据采集']):
        return 'Extractive'

    # 主体判断有结果就用主体判断
    if actor_class:
        return actor_class

    # 否则用内容判断
    if content_class:
        return content_class

    # 都没有则归为协调能力（政府日常运作）
    return 'Coordination'


def get_classification_reason(row, classification):
    """生成分类理由"""
    purchaser = str(row.get('采购人', ''))
    contract_name = str(row.get('合同名称', ''))

    if classification == 'Extractive':
        if any(kw in purchaser for kw in ['税务', '国税', '地税']):
            return '税务机关-增强税收征管能力'
        elif any(kw in purchaser for kw in ['海关']):
            return '海关-增强关税征收和边境管控'
        elif any(kw in purchaser for kw in ['财政']):
            return '财政部门-增强财政收入管理'
        elif any(kw in purchaser for kw in ['银行', '金融']):
            return '金融机构-增强金融监管/资源调配'
        elif any(kw in purchaser for kw in ['国土', '资源', '矿产']):
            return '资源部门-增强资源获取和管理'
        elif any(kw in contract_name for kw in ['监测', '调查', '数据采集', '遥感']):
            return '信息采集-增强资源/数据获取能力'
        else:
            return '增强国家资源获取能力'

    elif classification == 'Compliance':
        if any(kw in purchaser for kw in ['公安', '警察']):
            return '公安机关-增强执法和社会管控能力'
        elif any(kw in purchaser for kw in ['法院', '检察', '司法']):
            return '司法机关-增强司法执行能力'
        elif any(kw in purchaser for kw in ['监察', '纪委']):
            return '监察机关-增强反腐监督能力'
        elif any(kw in purchaser for kw in ['市场监管', '工商', '质监']):
            return '市场监管-增强市场合规监管能力'
        elif any(kw in purchaser for kw in ['环保', '生态']):
            return '环保部门-增强环境执法能力'
        elif any(kw in contract_name for kw in ['监控', '执法', '安防']):
            return '执法监控-增强合规监管能力'
        else:
            return '增强国家合规监管能力'

    else:  # Coordination
        if any(kw in purchaser for kw in ['学校', '中学', '小学', '大学', '学院', '教育']):
            return '教育机构-组织教育公共服务'
        elif any(kw in purchaser for kw in ['医院', '卫生', '疾控']):
            return '医疗机构-组织公共卫生服务'
        elif any(kw in purchaser for kw in ['科学院', '研究所', '研究院']):
            return '科研机构-组织科学研究'
        elif any(kw in purchaser for kw in ['交通', '公路', '铁路']):
            return '交通部门-组织交通基础设施'
        elif any(kw in purchaser for kw in ['文化', '博物', '图书馆']):
            return '文化机构-组织文化公共服务'
        elif any(kw in purchaser for kw in ['农业', '农牧', '林业']):
            return '农业部门-组织农业生产服务'
        else:
            return '组织公共服务和集体行动'


def main():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel('2015.xls')
    print(f"数据总量: {len(df)} 条")

    # 进行分类
    print("\n正在用LLM逻辑进行分类...")
    df['国家能力分类'] = df.apply(classify_state_capacity, axis=1)
    df['分类理由'] = df.apply(lambda row: get_classification_reason(row, row['国家能力分类']), axis=1)

    # 统计结果
    print("\n=== 分类结果统计 ===")
    print(df['国家能力分类'].value_counts())
    print(f"\n占比:")
    print(df['国家能力分类'].value_counts(normalize=True).round(3) * 100)

    # 展示各类别示例
    print("\n=== 各类别示例 ===")
    for cap_type in ['Extractive', 'Compliance', 'Coordination']:
        print(f"\n【{cap_type}】:")
        samples = df[df['国家能力分类'] == cap_type][['采购人', '合同名称', '分类理由']].head(5)
        for _, row in samples.iterrows():
            print(f"  • {row['采购人'][:20]} | {row['合同名称'][:30]} | {row['分类理由']}")

    # 保存结果
    output_file = '2015_classified.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n分类结果已保存到: {output_file}")

    return df


if __name__ == '__main__':
    main()
