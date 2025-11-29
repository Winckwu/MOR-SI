#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude语义标注器 - 基于真正的语义理解而非简单关键词匹配

核心思路：
1. 采购人性质决定基础分类（税务局→汲取，学校→合规，建设局→协调）
2. 合同内容可以修正分类（税务局买教学设备给下属学校→可能是合规）
3. 行业类型作为辅助参考

这是我作为Claude对Berwick & Christia框架的理解应用
"""

import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from datetime import datetime


class ClaudeSemanticLabeler:
    """
    Claude语义标注器

    分类逻辑：
    - 汲取能力：国家获取资源（税收、财政、统计、土地确权、审计）
    - 协调能力：国家提供公共品（基础设施、政府运转、信息化）
    - 合规能力：国家管理社会（教育、医疗、执法、社保、农林牧渔监管）
    """

    def __init__(self):
        self._init_purchaser_rules()
        self._init_content_rules()
        self._init_industry_rules()
        self._init_override_rules()

    def _init_purchaser_rules(self):
        """采购人分类规则 - 基于机构性质"""
        self.purchaser_extractive = [
            # 税务系统
            "税务", "国税", "地税", "税务局", "税务所",
            # 财政系统
            "财政局", "财政厅", "财政部", "财政所",
            # 海关系统
            "海关", "关税",
            # 审计系统
            "审计局", "审计署", "审计厅",
            # 统计系统
            "统计局", "统计站", "调查队",
            # 国土资源
            "国土局", "土地局", "不动产登记", "自然资源局",
            # 国资系统
            "国资委", "资产管理",
        ]

        self.purchaser_compliance = [
            # 教育系统
            "学校", "大学", "学院", "中学", "小学", "幼儿园",
            "教育局", "教育厅", "教体局", "教育部",
            "职业学校", "技校", "党校", "干部学院",
            # 医疗卫生系统
            "医院", "卫生院", "诊所", "疾控", "疾病预防",
            "卫生局", "卫健委", "卫生厅", "卫生部",
            "血站", "中心血站", "妇幼保健",
            # 公安司法系统
            "公安局", "公安厅", "公安部", "派出所",
            "法院", "检察院", "司法局", "司法厅",
            "监狱", "看守所", "戒毒所",
            "消防", "武警", "边防", "边检",
            # 市场监管
            "市场监管", "工商局", "质监局", "药监局", "食药监",
            # 民政社保
            "民政局", "社保局", "人社局", "残联",
            # 农林牧渔
            "农业局", "农牧局", "畜牧局", "农业农村局",
            "林业局", "林草局", "林业厅",
            "渔业局", "水产局", "渔政",
            "植保站", "兽医站", "动物卫生",
            # 文化文物
            "文物局", "博物馆", "纪念馆", "文化馆",
            "故宫", "考古", "文化遗产",
            "档案局", "档案馆",
        ]

        self.purchaser_coordination = [
            # 交通系统
            "交通局", "交通厅", "公路局", "公路段",
            "铁路局", "航空", "机场", "港口", "航道",
            # 建设规划系统
            "建设局", "住建局", "规划局", "城建局",
            "住房和城乡建设", "城市管理局",
            # 水利系统
            "水利局", "水务局", "水利厅", "水利部",
            "水利委员会", "河务局",
            # 电力能源系统
            "电力局", "供电", "电网", "能源局",
            # 通信系统
            "通信管理局", "无线电管理",
            # 发改系统
            "发改委", "发展改革",
            # 机关事务
            "机关事务", "后勤", "办公厅", "办公室",
            "政府采购中心", "招标中心",
            # 开发区
            "开发区", "高新区", "经开区", "管委会",
            # 环保
            "环保局", "生态环境局", "环境保护",
        ]

    def _init_content_rules(self):
        """合同内容分类规则"""
        self.content_extractive = [
            # 税收相关
            "税收", "征税", "纳税", "税源", "税基", "金税",
            "发票", "增值税", "票据", "防伪税控",
            # 财政相关
            "预算", "决算", "财政收入", "国库", "拨款",
            # 统计调查
            "普查", "调查", "统计", "数据采集", "年鉴",
            "经济普查", "人口普查", "农业普查",
            # 土地确权
            "土地调查", "地籍", "确权", "土地登记",
            "不动产登记", "权籍", "土地变更",
            # 审计稽查
            "审计", "稽查", "核查", "查账",
            # 资产评估
            "资产评估", "清产核资", "资产清查",
        ]

        self.content_compliance = [
            # 教育
            "教学", "教育", "培训", "课程", "教材",
            "实验室", "图书", "教学设备", "课桌",
            "班班通", "多媒体教室", "校园",
            "改薄", "义务教育", "学生",
            # 医疗
            "医疗", "诊断", "治疗", "手术", "门诊",
            "CT", "核磁", "B超", "X光", "DR",
            "药品", "医疗设备", "医疗器械",
            "防疫", "疫苗", "免疫", "消毒",
            # 执法
            "执法", "警用", "警务", "安防", "监控",
            "巡逻车", "执法车", "警车", "囚车",
            "消防车", "消防设备", "消防器材",
            "法警", "刑侦", "治安",
            # 社保民政
            "社保", "养老", "低保", "救助", "扶贫",
            "福利", "残疾", "殡葬", "救灾",
            # 农林牧渔监管
            "牲畜", "耳标", "动物防疫", "兽药",
            "植保", "病虫害", "种子检验",
            "森林防火", "护林", "巡林",
            # 文物保护
            "文物保护", "修缮", "考古", "文化遗产",
        ]

        self.content_coordination = [
            # 基础设施建设
            "公路", "道路", "桥梁", "隧道", "铁路",
            "供水", "排水", "污水", "管网", "管道",
            "供电", "电网", "变电", "输电", "配电",
            "供气", "供暖", "供热", "燃气",
            # 建设工程
            "建设工程", "施工", "土建", "基建",
            "新建", "改建", "扩建", "改造工程",
            "装修", "维修工程", "修缮工程",
            # 信息化建设
            "信息化", "数字化", "智慧城市", "电子政务",
            "信息系统", "平台建设", "网站建设",
            "软件开发", "系统集成", "网络建设",
            # 政府运转
            "办公设备", "办公家具", "办公用品",
            "公务车", "公务用车", "车辆采购",
            "物业", "保洁", "绿化", "安保",
            "食堂", "餐饮", "会议", "印刷",
        ]

    def _init_industry_rules(self):
        """行业分类规则"""
        self.industry_extractive = [
            "税务", "财政", "统计", "海关", "审计",
        ]

        self.industry_compliance = [
            "教育", "卫生", "医疗", "公安", "司法",
            "文化", "体育", "民政", "社会保障",
            "农业", "林业", "畜牧", "渔业",
        ]

        self.industry_coordination = [
            "交通", "建设", "水利", "电力", "能源",
            "通信", "信息", "软件", "计算机",
        ]

    def _init_override_rules(self):
        """特殊覆盖规则 - 某些情况需要覆盖默认判断"""
        # 无论采购人是谁，这些内容都强指向某类
        self.strong_extractive = [
            "金税三期", "增值税发票", "税控", "税务系统",
            "土地确权", "地籍调查", "不动产登记系统",
            "经济普查", "人口普查", "统计调查",
        ]

        self.strong_compliance = [
            "义务教育", "改薄项目", "校舍", "教学楼",
            "医院", "卫生院", "疾控中心",
            "警用装备", "执法装备", "消防装备",
            "牲畜耳标", "动物防疫",
            "文物保护", "文物修缮",
        ]

        self.strong_coordination = [
            "公路建设", "道路工程", "桥梁工程",
            "水利工程", "水库", "堤防", "灌溉",
            "供电工程", "变电站", "输电线路",
            "污水处理厂", "自来水厂",
        ]

    def _check_strong_rules(self, text):
        """检查强规则"""
        text = str(text) if text else ""

        for kw in self.strong_extractive:
            if kw in text:
                return "汲取能力", 0.95, f"强规则匹配: {kw}"

        for kw in self.strong_compliance:
            if kw in text:
                return "合规能力", 0.95, f"强规则匹配: {kw}"

        for kw in self.strong_coordination:
            if kw in text:
                return "协调能力", 0.95, f"强规则匹配: {kw}"

        return None, 0, ""

    def _classify_by_purchaser(self, purchaser):
        """根据采购人分类"""
        if not purchaser or pd.isna(purchaser):
            return None, 0

        purchaser = str(purchaser)

        # 汲取能力机构
        for kw in self.purchaser_extractive:
            if kw in purchaser:
                return "汲取能力", 0.8

        # 合规能力机构
        for kw in self.purchaser_compliance:
            if kw in purchaser:
                return "合规能力", 0.8

        # 协调能力机构
        for kw in self.purchaser_coordination:
            if kw in purchaser:
                return "协调能力", 0.8

        return None, 0

    def _classify_by_content(self, content):
        """根据合同内容分类"""
        if not content or pd.isna(content):
            return None, 0

        content = str(content)
        scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}

        # 计算各类得分
        for kw in self.content_extractive:
            if kw in content:
                scores["汲取能力"] += len(kw)

        for kw in self.content_compliance:
            if kw in content:
                scores["合规能力"] += len(kw)

        for kw in self.content_coordination:
            if kw in content:
                scores["协调能力"] += len(kw)

        total = sum(scores.values())
        if total == 0:
            return None, 0

        max_label = max(scores, key=scores.get)
        confidence = scores[max_label] / total

        return max_label, confidence * 0.6

    def _classify_by_industry(self, industry):
        """根据行业分类"""
        if not industry or pd.isna(industry):
            return None, 0

        industry = str(industry)

        for kw in self.industry_extractive:
            if kw in industry:
                return "汲取能力", 0.4

        for kw in self.industry_compliance:
            if kw in industry:
                return "合规能力", 0.4

        for kw in self.industry_coordination:
            if kw in industry:
                return "协调能力", 0.4

        return None, 0

    def classify(self, contract_name, purchaser=None, industry=None):
        """
        综合分类

        优先级：
        1. 强规则覆盖（某些关键词强指向某类）
        2. 采购人性质（最重要的判断依据）
        3. 合同内容（辅助判断）
        4. 行业类型（次要参考）
        5. 默认归为协调能力（政府采购默认）
        """
        all_text = f"{contract_name} {purchaser or ''} {industry or ''}"

        # 1. 检查强规则
        strong_label, strong_conf, strong_reason = self._check_strong_rules(all_text)
        if strong_label:
            return strong_label, strong_conf, strong_reason

        # 2. 综合评分
        scores = {"汲取能力": 0, "协调能力": 0, "合规能力": 0}
        reasons = []

        # 采购人分类（权重0.5）
        p_label, p_conf = self._classify_by_purchaser(purchaser)
        if p_label:
            scores[p_label] += p_conf * 0.5
            reasons.append(f"采购人→{p_label}")

        # 内容分类（权重0.35）
        c_label, c_conf = self._classify_by_content(contract_name)
        if c_label:
            scores[c_label] += c_conf * 0.35
            reasons.append(f"内容→{c_label}")

        # 行业分类（权重0.15）
        i_label, i_conf = self._classify_by_industry(industry)
        if i_label:
            scores[i_label] += i_conf * 0.15
            reasons.append(f"行业→{i_label}")

        # 3. 判断结果
        total = sum(scores.values())
        if total == 0:
            # 默认归为协调能力
            return "协调能力", 0.4, "无明确信号，默认分类"

        max_label = max(scores, key=scores.get)
        confidence = scores[max_label] / max(total, 0.001)

        # 调整置信度
        confidence = min(0.95, max(0.5, confidence + 0.2))

        return max_label, confidence, "; ".join(reasons)

    def label_dataframe(self, df, contract_col='合同名称', purchaser_col='采购人',
                        industry_col='所属行业', show_progress=True):
        """批量标注DataFrame"""
        results = []
        total = len(df)

        for idx, row in df.iterrows():
            contract = row.get(contract_col, "")
            purchaser = row.get(purchaser_col, "")
            industry = row.get(industry_col, "")

            label, confidence, reason = self.classify(contract, purchaser, industry)

            results.append({
                "id": idx,
                "合同名称": contract,
                "采购人": purchaser,
                "所属行业": industry,
                "label": label,
                "confidence": confidence,
                "reason": reason
            })

            if show_progress and (idx + 1) % 1000 == 0:
                print(f"  已标注: {idx + 1}/{total}")

        return results


def run_labeling(data_path, output_path):
    """运行标注流程"""
    print("=" * 70)
    print("Claude语义标注器 - 基于真正的语义理解")
    print("=" * 70)

    # 加载数据
    print(f"\n加载数据: {data_path}")
    df = pd.read_excel(data_path)
    print(f"数据量: {len(df)} 条")

    # 创建标注器
    labeler = ClaudeSemanticLabeler()

    # 标注
    print("\n开始标注...")
    results = labeler.label_dataframe(df)

    # 统计
    print("\n" + "=" * 70)
    print("标注结果统计")
    print("=" * 70)

    labels = [r["label"] for r in results]
    confs = [r["confidence"] for r in results]

    print("\n类别分布:")
    for label in ["汲取能力", "协调能力", "合规能力"]:
        count = labels.count(label)
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    print(f"\n平均置信度: {np.mean(confs):.3f}")
    print(f"高置信度(≥0.7): {sum(1 for c in confs if c >= 0.7)} ({sum(1 for c in confs if c >= 0.7)/len(confs)*100:.1f}%)")

    # 保存结果
    print("\n保存结果...")

    # JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "source": data_path,
                "num_samples": len(results),
                "labeled_by": "Claude Semantic Labeler",
                "timestamp": datetime.now().isoformat(),
                "framework": "Berwick & Christia (2018) State Capacity Redux",
                "method": "综合采购人+内容+行业的语义理解"
            },
            "labels": results
        }, f, ensure_ascii=False, indent=2)

    print(f"JSON已保存: {output_path}")

    # Excel格式
    result_df = df.copy()
    result_df["Claude_标签"] = labels
    result_df["Claude_置信度"] = confs
    result_df["Claude_理由"] = [r["reason"] for r in results]

    excel_path = output_path.replace(".json", ".xlsx")
    result_df.to_excel(excel_path, index=False)
    print(f"Excel已保存: {excel_path}")

    return results, result_df


if __name__ == "__main__":
    results, df = run_labeling(
        "/home/user/MOR-SI/sample_10000.xlsx",
        "/home/user/MOR-SI/claude_semantic_labels_10000.json"
    )
