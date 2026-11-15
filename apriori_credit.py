#!/usr/bin/env python3
"""
信用卡客户违约关联规则分析 (Apriori 算法自行实现)
- 候选项集生成、支持度统计、剪枝均为手写，不调用现成 Apriori 库
- 后件固定为 R(违约)，前件含 1~3 个离散特征
- 最小支持度 / 置信度 / 提升度 可通过命令行参数设置
- 仅使用 CSV 中 split=='train' 的拟合集记录，不改变原始划分

用法:
  python apriori_credit.py
  python apriori_credit.py --min-support 0.05 --min-confidence 0.50 --min-lift 1.20
  python apriori_credit.py --csv "路径" --min-support 0.10 --min-confidence 0.80 --min-lift 1.50
"""

import sys
import os
import time
import argparse
from itertools import combinations

# 解决 Windows 控制台中文输出乱码
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 尝试用 pandas 读数据，不可用则回退到 csv 模块
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# 共享离散化与路径解析逻辑
import credit_common as cm

TARGET = cm.TARGET          # 后件项名: R 代表违约
DEFAULT_CSV = cm.DEFAULT_CSV  # 相对路径，可在脚本目录或工作目录；亦可用 --csv 指定


# ==================== 数据预处理 ====================
def load_transactions(csv_path, split_col="train"):
    """读取 CSV，仅保留 split 列匹配的记录，调用共享模块离散化为项集(frozenset)"""
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        total_all = len(df)
        df = df[df["split"] == split_col].copy()
        print(f"  按 split='{split_col}' 筛选: {total_all:,} → {len(df):,} 条")
    transactions, _ = cm.discretize(df)
    return transactions, len(df)


# ==================== Apriori 核心算法 ====================
def apriori_gen(L_prev, k):
    """
    候选项集生成：由 (k-1)-频繁项集自连接生成 k-候选项集，再用先验性质剪枝
    - 自连接: 前 k-2 项相同的两个项集合并
    - 剪枝:   候选的任一 (k-1)-子集必须都是频繁的
    """
    prev_list = list(L_prev.keys())
    candidates = set()
    n = len(prev_list)

    for i in range(n):
        for j in range(i + 1, n):
            p = prev_list[i]
            q = prev_list[j]
            # 排序后比较前 k-2 个元素是否相同
            p_sorted = sorted(p)
            q_sorted = sorted(q)
            if p_sorted[:k - 2] == q_sorted[:k - 2]:
                candidate = frozenset(p | q)  # 合并两个项集
                if len(candidate) != k:
                    continue
                # 剪枝：检查所有 (k-1)-子集是否都在 L_prev 中
                all_sub_frequent = True
                for sub in combinations(candidate, k - 1):
                    if frozenset(sub) not in L_prev:
                        all_sub_frequent = False
                        break
                if all_sub_frequent:
                    candidates.add(candidate)

    return candidates


def count_support(transactions, candidates, k, item_tx=None):
    """
    支持度统计
    - 优先用倒排索引交集法: 候选的支持度 = 各项"事务索引集合"的交集大小
      复杂度约 O(候选数 × k × 事务数/位宽)，远快于逐事务生成 C(|t|,k) 子集
    - 无倒排索引时回退到逐事务生成 k-子集查表
    """
    counts = dict.fromkeys(candidates, 0)
    if not candidates:
        return counts
    if item_tx is not None:
        for c in candidates:
            # 取各项的事务索引集合，从最小集合开始交集(提前剪枝)
            idx_sets = [item_tx[it] for it in c]
            idx_sets.sort(key=len)
            common = idx_sets[0]
            for s in idx_sets[1:]:
                common = common & s
                if not common:
                    break
            counts[c] = len(common)
        return counts
    # 回退: 逐事务生成 k-子集查表
    for t in transactions:
        if len(t) < k:
            continue
        for sub in combinations(t, k):
            fs = frozenset(sub)
            if fs in counts:
                counts[fs] += 1
    return counts


def apriori(transactions, min_support, max_k):
    """
    Apriori 主算法：
    1. 扫描数据库统计 1-项集支持度，剪枝得到 L1
    2. 迭代: 候选生成 → 支持度统计 → 剪枝，得到 Lk
    3. 返回所有频繁项集 {项集: 支持度计数}
    """
    total = len(transactions)
    min_count = min_support * total  # 最小支持度计数

    # 预计算倒排索引: item -> 事务索引集合(加速支持度统计)
    item_tx = {}
    for i, t in enumerate(transactions):
        for item in t:
            item_tx.setdefault(item, set()).add(i)

    # --- 第一轮：统计所有 1-项集 ---
    item_counts = {item: len(idx) for item, idx in item_tx.items()}

    # 剪枝：保留支持度达标的 1-项集
    L_prev = {}
    for item, cnt in item_counts.items():
        if cnt >= min_count:
            L_prev[frozenset([item])] = cnt

    all_frequent = dict(L_prev)  # 汇总所有频繁项集

    # --- 迭代生成 k-项集 (k=2,3,...,max_k) ---
    k = 2
    while L_prev and k <= max_k:
        Ck = apriori_gen(L_prev, k)       # 候选生成（自连接 + 剪枝）
        if not Ck:
            break
        candidate_counts = count_support(transactions, Ck, k, item_tx)  # 支持度统计
        L_curr = {c: cnt for c, cnt in candidate_counts.items() if cnt >= min_count}  # 剪枝
        all_frequent.update(L_curr)
        L_prev = L_curr
        k += 1

    return all_frequent, total


# ==================== 规则生成 ====================
def generate_rules(frequent, total, min_sup, min_conf, min_lift, target, max_ante):
    """
    从频繁项集中生成关联规则：
    - 后件固定为 {target}
    - 前件 = 频繁项集 - {target}，大小 1~max_ante
    - 置信度 = support(前件∪后件) / support(前件)
    - 提升度 = 置信度 / support(后件)
    """
    target_set = frozenset([target])
    support_target = frequent.get(target_set, 0) / total  # 后件支持度

    if support_target == 0:
        return []

    rules = []
    for itemset, count in frequent.items():
        if target not in itemset or len(itemset) < 2:
            continue

        antecedent = itemset - target_set  # 前件
        ante_len = len(antecedent)
        if ante_len < 1 or ante_len > max_ante:
            continue

        ante_count = frequent.get(antecedent, 0)
        if ante_count == 0:
            continue

        # 计算指标
        support = count / total              # 规则支持度
        confidence = count / ante_count      # 置信度
        lift = confidence / support_target   # 提升度

        # 过滤：支持度、置信度、提升度同时达标
        if support >= min_sup and confidence >= min_conf and lift >= min_lift:
            rules.append({
                "antecedent": antecedent,
                "hit_count": count,        # 命中次数(前件∧后件)
                "ante_count": ante_count,  # 前件命中次数
                "support": support,
                "confidence": confidence,
                "lift": lift,
            })

    # 按提升度降序排列
    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules


# ==================== 输出格式化 ====================
def format_itemset(itemset):
    """格式化项集为可读字符串"""
    return "{" + ", ".join(sorted(itemset)) + "}"


def print_rules_section(title, rules, show_limit=30):
    """输出一个规则表格"""
    print("\n" + "-" * 72)
    print(f"  {title}")
    print("-" * 72)
    if not rules:
        print("  （无满足条件的规则）")
        return
    print(f"  {'规则':<44s} {'命中':>6s} {'支持度':>8s} {'置信度':>8s} {'提升度':>8s}")
    print("  " + "-" * 68)
    shown = 0
    for r in rules:
        if shown >= show_limit:
            break
        ante_str = format_itemset(r["antecedent"])
        rule_str = f"{ante_str} → {{R}}"
        print(f"  {rule_str:<44s} {r['hit_count']:>6d} {r['support']:>8.4f} "
              f"{r['confidence']:>8.4f} {r['lift']:>8.4f}")
        shown += 1
    more = len(rules) - shown
    if more > 0:
        print(f"  ... 还有 {more} 条未显示")
    print(f"\n  共 {len(rules)} 条规则")


def print_results(frequent, rules, ref_rules, total, elapsed, args):
    """输出频繁项集和关联规则"""

    target_set = frozenset([TARGET])
    sup_r = frequent.get(target_set, 0) / total

    print("\n" + "=" * 72)
    print("  信用卡客户违约关联规则分析结果 (拟合集 train)")
    print("=" * 72)
    print(f"  总记录数: {total:,}    违约(R)支持度: {sup_r:.4f} ({frequent.get(target_set,0):,}条)")
    print(f"  最小支持度: {args.min_support}  最小置信度: {args.min_confidence}  最小提升度: {args.min_lift}")
    print(f"  后件: {{R}} (R=违约)  前件特征数: 1~{args.max_antecedent}")
    print(f"  耗时: {elapsed:.2f} 秒")

    # --- 输出频繁项集 ---
    print("\n" + "-" * 72)
    print("  一、频繁项集")
    print("-" * 72)
    print(f"  {'项数':>4s}  {'支持度':>8s}  {'命中次数':>8s}  项集")
    print("  " + "-" * 68)

    by_size = {}
    for itemset, count in frequent.items():
        by_size.setdefault(len(itemset), []).append((itemset, count))

    fi_count = 0
    for size in sorted(by_size.keys()):
        for itemset, count in sorted(by_size[size], key=lambda x: -x[1]):
            print(f"  {size:>4d}  {count / total:>8.4f}  {count:>8d}  {format_itemset(itemset)}")
            fi_count += 1
    print(f"\n  频繁项集总数: {fi_count}")

    # --- 输出关联规则（严格阈值）---
    print_rules_section(
        f"二、关联规则 (后件→R, 支持度≥{args.min_support}, 置信度≥{args.min_confidence}, 提升度≥{args.min_lift})",
        rules
    )
    if not rules:
        print(f"  注: 违约率仅 {sup_r:.1%}，单一特征组合难以达到 {args.min_confidence:.0%} 置信度")

    # --- 输出参考规则（放宽阈值）---
    if ref_rules and not rules:
        print_rules_section(
            "三、参考规则 (放宽: 置信度≥0.30, 提升度≥1.20)",
            ref_rules
        )

    print("\n" + "=" * 72)


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="信用卡违约关联规则分析 (Apriori)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV 文件路径")
    parser.add_argument("--min-support", type=float, default=0.10, help="最小支持度 (默认 0.10)")
    parser.add_argument("--min-confidence", type=float, default=0.80, help="最小置信度 (默认 0.80)")
    parser.add_argument("--min-lift", type=float, default=1.50, help="最小提升度 (默认 1.50)")
    parser.add_argument("--max-antecedent", type=int, default=3, help="前件最大特征数 (默认 3)")
    parser.add_argument("--split", default="train", help="使用的数据划分 (默认 train)")
    args = parser.parse_args()

    csv_path = cm.resolve_csv(args.csv)
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 → {args.csv}（已尝试工作目录与脚本目录）")
        sys.exit(1)

    max_k = args.max_antecedent + 1  # 前件 + 后件 R

    print(f"正在加载数据: {csv_path}")
    t0 = time.time()
    transactions, total = load_transactions(csv_path, args.split)
    t1 = time.time()
    print(f"数据加载完成: {total:,} 条记录（split={args.split}），耗时 {t1 - t0:.2f} 秒")

    print("正在运行 Apriori 算法...")
    frequent, total = apriori(transactions, args.min_support, max_k)
    t2 = time.time()
    print(f"频繁项集挖掘完成: {len(frequent)} 个，耗时 {t2 - t1:.2f} 秒")

    # 严格阈值规则
    rules = generate_rules(frequent, total, args.min_support, args.min_confidence,
                           args.min_lift, TARGET, args.max_antecedent)
    # 参考规则（放宽阈值，用于无严格规则时展示数据中的实际模式）
    ref_rules = generate_rules(frequent, total, args.min_support, 0.30, 1.20,
                               TARGET, args.max_antecedent) if not rules else []

    print_results(frequent, rules, ref_rules, total, t2 - t0, args)


if __name__ == "__main__":
    main()