===================================
政府采购合同国家能力分类器
===================================

【环境要求】
pip install pandas scikit-learn

【使用方法】
python classify_with_model.py 2016.csv
python classify_with_model.py 2017.dta
python classify_with_model.py 2018.xlsx

【输出文件】
原文件名_classified.csv

【分类类别】
- 提取能力：税务局、海关、统计局
- 合规能力：公安、检察院、法院、市场监管
- 协调能力：教育、医疗、水利、交通

【批量处理】
for f in 2016.csv 2017.csv 2018.csv; do
    python classify_with_model.py $f
done
