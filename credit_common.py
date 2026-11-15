#!/usr/bin/env python3
"""
信用卡违约数据共享模块
- 统一离散化逻辑(供 apriori_credit.py 与 logistic_credit_v2.py 复用)
- 统一 CSV 路径解析(相对路径，跨机器可移植)

离散化规则:
  - 额度/年龄: 三分位
  - 性别/婚姻: 二分类
  - 学历: 保留原始类别(研究生/本科/高中/其他学历/学历未知)
  - 各月还款状态(6个月): 按时/延迟
  - 6个月最大延迟: 无延迟/有延迟
  - 各月账单金额(6个月): 低/高(按 train 中位数)
  - 各月已还金额(6个月): 低/高(按 train 中位数)
  - 违约标签: R
"""

import os

import pandas as pd

DEFAULT_CSV = "01_credit_default.csv"  # 相对路径，可在脚本目录或工作目录；亦可用 --csv 指定
TARGET = "R"                            # 后件项名: R 代表违约
MONTHS = ["09", "08", "07", "06", "05", "04"]


def resolve_csv(path):
    """解析 CSV 路径: 优先按给定路径查找，找不到则回退到脚本所在目录"""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path  # 返回原路径，由后续存在性检查报错


def discretize(df, q=None):
    """
    将 DataFrame 每行离散化为项集(frozenset)
    - q 为分位数字典，由 train 集计算；valid/test 复用同一 q 避免信息泄露
    - 返回 (transactions, q)
    """
    pay_cols = [f"pay_status_2005_{m}" for m in MONTHS]
    if q is None:
        q = {
            "lq33": df["credit_limit_ntd"].quantile(0.33),
            "lq67": df["credit_limit_ntd"].quantile(0.67),
            "aq33": df["age_years"].quantile(0.33),
            "aq67": df["age_years"].quantile(0.67),
            "bill_med": {m: df[f"bill_amount_2005_{m}_ntd"].median() for m in MONTHS},
            "paid_med": {m: df[f"paid_amount_2005_{m}_ntd"].median() for m in MONTHS},
        }
    txs = []
    for i in range(len(df)):
        r = df.iloc[i]
        s = set()
        # 额度三分位
        lim = float(r["credit_limit_ntd"])
        s.add("额度低" if lim < q["lq33"] else ("额度中" if lim < q["lq67"] else "额度高"))
        # 年龄三分位
        age = float(r["age_years"])
        s.add("年轻" if age < q["aq33"] else ("中年" if age < q["aq67"] else "年长"))
        # 性别
        s.add("男" if int(r["sex_code"]) == 1 else "女")
        # 学历(保留原始类别，避免高中被并入低学历)
        edu = int(r["education_code"])
        if edu == 1:
            s.add("研究生")
        elif edu == 2:
            s.add("本科")
        elif edu == 3:
            s.add("高中")
        elif edu in (4, 5, 6):
            s.add("其他学历")
        else:
            s.add("学历未知")
        # 婚姻
        s.add("已婚" if int(r["marriage_code"]) == 1 else "未婚")
        # 各月还款状态(6个月)
        for m in MONTHS:
            ps = int(r[f"pay_status_2005_{m}"])
            s.add(f"{m}月按时" if ps <= 0 else f"{m}月延迟")
        # 6个月最大延迟
        max_delay = max(int(r[c]) for c in pay_cols)
        s.add("无延迟" if max_delay <= 0 else "有延迟")
        # 各月账单金额 低/高(按 train 中位数)
        for m in MONTHS:
            bill = float(r[f"bill_amount_2005_{m}_ntd"])
            s.add(f"{m}月账单低" if bill < q["bill_med"][m] else f"{m}月账单高")
        # 各月已还金额 低/高(按 train 中位数)
        for m in MONTHS:
            paid = float(r[f"paid_amount_2005_{m}_ntd"])
            s.add(f"{m}月还款低" if paid < q["paid_med"][m] else f"{m}月还款高")
        # 违约标签
        if int(r["default_next_month"]) == 1:
            s.add(TARGET)
        txs.append(frozenset(s))
    return txs, q