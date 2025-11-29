"""
用 Claude 的理解逻辑对所有采购合同进行国家能力分类
结合 主体类型 + 采购目的 综合判断
"""

import pandas as pd
import re

def llm_classify(purchaser, contract_name):
    """
    模拟 Claude 的理解逻辑进行分类

    分类维度：
    1. 提取能力 (Extractive) - 国家获取资源的能力
    2. 协调能力 (Coordination) - 组织集体行动的能力
    3. 合规能力 (Compliance) - 确保公众和官员遵从的能力
    """

    purchaser = str(purchaser) if pd.notna(purchaser) else ''
    contract_name = str(contract_name) if pd.notna(contract_name) else ''

    combined = purchaser + contract_name

    # ============ 合规能力 (Compliance) ============
    # 执法、司法、监察、监管类机构
    compliance_actors = [
        '公安', '警察', '交警', '刑警', '治安', '派出所',
        '法院', '检察', '司法', '监狱', '戒毒', '看守所',
        '纪委', '监委', '监察', '巡视', '巡察',
        '市场监管', '工商', '质监', '药监', '食药', '卫生监督',
        '环保', '生态环境', '环境监察', '环境执法',
        '安监', '应急', '消防', '武警', '边防', '海警',
        '城管', '综合执法', '行政执法',
        '海事', '渔政', '林政', '国土执法', '矿山执法'
    ]

    compliance_purposes = [
        '监控', '监视', '安防', '安保', '警用', '警务',
        '执法', '取证', '审讯', '讯问', '勘查',
        '巡逻', '巡查', '稽查', '检查', '抽检',
        '处罚', '强制', '拘留', '羁押',
        '鉴定', '检测', '检验', '司法鉴定'
    ]

    for kw in compliance_actors:
        if kw in purchaser:
            # 确认是执法/监管类采购
            reason = get_compliance_reason(purchaser, contract_name)
            return '合规能力', reason

    for kw in compliance_purposes:
        if kw in contract_name:
            reason = f'增强{kw}能力'
            return '合规能力', reason

    # ============ 提取能力 (Extractive) ============
    # 税收、财政、资源获取、信息采集类
    extractive_actors = [
        '税务', '国税', '地税', '税收',
        '海关', '关税',
        '财政', '国库', '金库',
        '银行', '人民银行', '央行', '金融', '证监', '银监', '保监',
        '国土', '自然资源', '土地', '矿产', '地质',
        '统计', '普查', '调查队',
        '审计', '会计',
        '能源', '电力', '石油', '煤炭', '天然气',
        '外汇', '国资'
    ]

    extractive_purposes = [
        '征收', '征管', '税收', '收费', '罚没',
        '监测', '遥感', '勘探', '勘查', '测量', '测绘', '普查', '调查',
        '数据采集', '信息采集', '信息系统', '数据平台', '管理平台',
        '登记', '确权', '不动产', '房产登记',
        '核算', '财务', '审计'
    ]

    for kw in extractive_actors:
        if kw in purchaser:
            reason = get_extractive_reason(purchaser, contract_name)
            return '提取能力', reason

    for kw in extractive_purposes:
        if kw in contract_name:
            # 但要排除教育机构的"信息系统"等
            if not any(edu in purchaser for edu in ['学校', '学院', '大学', '中学', '小学', '教育']):
                reason = f'增强{kw}能力'
                return '提取能力', reason

    # ============ 协调能力 (Coordination) ============
    # 教育、医疗、科研、基础设施、公共服务类
    coordination_actors = [
        '学校', '中学', '小学', '大学', '学院', '教育', '教体', '职业学校', '幼儿园',
        '医院', '卫生', '疾控', '疾病预防', '医疗', '卫健', '计生',
        '科学院', '研究所', '研究院', '科技', '科研',
        '交通', '公路', '铁路', '民航', '港口', '航道', '运输',
        '水利', '水务', '河道', '防汛',
        '农业', '农牧', '农机', '畜牧', '渔业', '林业', '草原',
        '文化', '文体', '博物', '图书馆', '档案', '广播', '电视', '广电', '新闻', '出版',
        '民政', '社保', '人社', '劳动', '就业', '养老', '残联',
        '住建', '规划', '房管', '城建', '市政',
        '旅游', '体育', '文旅',
        '气象', '地震', '测绘', '海洋研究',
        '党校', '行政学院', '干部学院'
    ]

    coordination_purposes = [
        '教学', '培训', '实训', '实验', '教育', '课程', '图书', '教材',
        '医疗', '诊断', '治疗', '康复', '护理', '防疫', '疫苗',
        '建设', '工程', '改造', '修缮', '维修', '装修',
        '科研', '研发', '实验', '试验', '课题',
        '服务', '保障', '管理', '办公', '会议', '后勤', '物业',
        '取暖', '空调', '家具', '设备', '车辆',
        '宣传', '文化', '保护', '遗产'
    ]

    for kw in coordination_actors:
        if kw in purchaser:
            reason = get_coordination_reason(purchaser, contract_name)
            return '协调能力', reason

    # 默认归为协调能力（政府日常运作）
    return '协调能力', '组织公共服务'


def get_compliance_reason(purchaser, contract_name):
    """生成合规能力分类理由"""
    if '公安' in purchaser or '警' in purchaser:
        if '消防' in purchaser:
            return '增强消防执法能力'
        if '交警' in purchaser or '交通' in purchaser:
            return '增强交通执法管控'
        return '增强执法机构能力'
    if '法院' in purchaser:
        return '增强司法审判能力'
    if '检察' in purchaser:
        return '增强检察监督能力'
    if '监察' in purchaser or '纪委' in purchaser:
        return '增强纪检监察能力'
    if '市场监管' in purchaser or '工商' in purchaser or '质监' in purchaser:
        return '增强市场监管能力'
    if '环保' in purchaser or '生态' in purchaser:
        return '增强环境执法能力'
    if '城管' in purchaser:
        return '增强城市执法能力'
    if '司法' in purchaser:
        return '增强司法行政能力'
    return '增强合规监管能力'


def get_extractive_reason(purchaser, contract_name):
    """生成提取能力分类理由"""
    if '税务' in purchaser or '国税' in purchaser or '地税' in purchaser:
        return '增强税收征管能力'
    if '海关' in purchaser:
        return '增强关税征收能力'
    if '财政' in purchaser:
        return '增强财政管理能力'
    if '银行' in purchaser or '金融' in purchaser:
        return '增强金融监管能力'
    if '国土' in purchaser or '自然资源' in purchaser or '土地' in purchaser:
        return '获取土地资源信息'
    if '统计' in purchaser:
        return '采集统计数据信息'
    if '审计' in purchaser:
        return '增强审计监督能力'
    if '房' in purchaser and ('管理' in purchaser or '登记' in purchaser):
        return '采集房产登记信息'
    if '能源' in purchaser or '电力' in purchaser:
        return '增强能源管理能力'
    if '监测' in contract_name or '调查' in contract_name:
        return '采集监测调查数据'
    if '信息系统' in contract_name or '平台' in contract_name:
        return '增强信息采集能力'
    return '增强资源获取能力'


def get_coordination_reason(purchaser, contract_name):
    """生成协调能力分类理由"""
    if any(kw in purchaser for kw in ['学校', '中学', '小学', '大学', '学院', '职业']):
        if '教学' in contract_name or '实训' in contract_name or '设备' in contract_name:
            return '组织职业教育服务'
        if '取暖' in contract_name or '空调' in contract_name:
            return '保障学校教学运转'
        return '组织教育公共服务'
    if '教育' in purchaser or '教体' in purchaser:
        return '组织教育公共服务'
    if '医院' in purchaser:
        return '组织公共医疗服务'
    if '卫生' in purchaser or '疾控' in purchaser:
        return '组织公共卫生服务'
    if '科学院' in purchaser or '研究所' in purchaser or '研究院' in purchaser:
        return '组织科学研究'
    if '交通' in purchaser or '公路' in purchaser:
        return '组织交通基础设施'
    if '农' in purchaser or '牧' in purchaser or '林' in purchaser:
        if '沼气' in contract_name or '能源' in contract_name:
            return '组织农村能源建设'
        if '退耕' in contract_name or '造林' in contract_name:
            return '组织生态农业项目'
        return '组织农业技术推广'
    if '文化' in purchaser or '文体' in purchaser or '广电' in purchaser:
        if '遗产' in contract_name or '保护' in contract_name:
            return '组织文化保护服务'
        return '组织宣传公共服务'
    if '民政' in purchaser or '社保' in purchaser:
        return '组织社会保障服务'
    if '住建' in purchaser or '城建' in purchaser:
        return '组织城市基础建设'
    if '水利' in purchaser or '水务' in purchaser:
        return '组织水利基础设施'
    if '党校' in purchaser:
        return '组织干部培训教育'
    if '机关事务' in purchaser:
        if '司法' in contract_name:
            return '增强司法机构能力'
        return '组织政府机关运转'
    if '互助会' in purchaser or '保障' in purchaser:
        if '信息系统' in contract_name:
            return '采集职工社保信息'
    return '组织公共服务'


def main():
    print("正在读取数据...")
    df = pd.read_excel('2015.xls')
    print(f"数据总量: {len(df)} 条")

    print("\n正在用 Claude 逻辑分类...")

    results = []
    for idx, row in df.iterrows():
        purchaser = row.get('采购人', '')
        contract_name = row.get('合同名称', '')

        classification, reason = llm_classify(purchaser, contract_name)
        results.append({
            '分类': classification,
            '理由': reason
        })

        if (idx + 1) % 5000 == 0:
            print(f"  已处理 {idx + 1} 条...")

    df['分类'] = [r['分类'] for r in results]
    df['理由'] = [r['理由'] for r in results]

    # 统计
    print("\n=== 分类结果统计 ===")
    print(df['分类'].value_counts())
    print("\n占比:")
    print((df['分类'].value_counts(normalize=True) * 100).round(1))

    # 保存
    output_file = '2015_state_capacity_classified.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")

    # 展示各类别示例
    print("\n=== 各类别示例 ===")
    for cap_type in ['提取能力', '合规能力', '协调能力']:
        print(f"\n【{cap_type}】:")
        samples = df[df['分类'] == cap_type][['采购人', '合同名称', '理由']].head(5)
        for _, row in samples.iterrows():
            p = str(row['采购人'])[:15]
            c = str(row['合同名称'])[:25]
            r = row['理由']
            print(f"  {p} | {c} | {r}")


if __name__ == '__main__':
    main()
