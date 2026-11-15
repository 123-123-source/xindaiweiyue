#!/usr/bin/env python3
"""
基于 apriori_credit.py 关联规则的信用卡违约逻辑回归模型 (重实现)
1. 复用 apriori_credit.py 中的 Apriori 算法与规则生成函数挖掘关联规则
2. 以规则前件作为 0/1 指示特征，train 集拟合逻辑回归(梯度下降 + L2)，valid 集验证
3. 阈值在 valid 集上按 F1 选出，test 集仅做一次性评价
4. 末尾加载 test 集，用锁定模型与阈值做最终评价

用法:
  python logistic_credit_v2.py
  python logistic_credit_v2.py --csv "路径" --n-features 20
"""

import sys
import os
import time
import argparse

# 解决 Windows 控制台中文乱码
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# 复用 apriori_credit.py 中的关联规则算法
import apriori_credit as ap
# 共享离散化与路径解析逻辑
import credit_common as cm

DEFAULT_CSV = cm.DEFAULT_CSV
TARGET = cm.TARGET
MONTHS = cm.MONTHS
resolve_csv = cm.resolve_csv
discretize = cm.discretize


# ==================== 特征选择与构建 ====================
def select_features(rules, n=15):
    """按提升度选取非冗余前件(当前前件不是已选前件的子集)"""
    selected = []
    for r in rules:
        ante = r["antecedent"]
        if any(ante <= s["antecedent"] for s in selected):
            continue
        selected.append(r)
        if len(selected) >= n:
            break
    return selected


def build_matrix(txs, features):
    """构建 0/1 特征矩阵: 记录是否匹配每个规则前件"""
    n = len(txs)
    m = len(features)
    X = np.zeros((n, m), dtype=np.float64)
    for i, t in enumerate(txs):
        for j, f in enumerate(features):
            if f["antecedent"] <= t:
                X[i, j] = 1.0
    return X


def dedup_features(X):
    """去除匹配模式完全相同的冗余特征列，返回保留列索引"""
    keep_idx = []
    seen = set()
    for j in range(X.shape[1]):
        key = bytes(X[:, j].astype(np.uint8))
        if key not in seen:
            seen.add(key)
            keep_idx.append(j)
    return keep_idx


# ==================== 逻辑回归 (梯度下降 + L2 正则 + 早停) ====================
def sigmoid(z):
    """数值稳定的 sigmoid"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_fit(X, y, lr=0.02, n_iter=20000, l2=1.0, tol=1e-7):
    """
    拟合逻辑回归: 梯度下降 + L2 正则
    - 学习率 lr=0.02 在收敛速度与稳定性间平衡
    - 损失变化小于 tol 时提前停止(收敛检查)
    """
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    eps = 1e-12
    prev_loss = float("inf")
    for it in range(n_iter):
        z = X @ w + b
        p = sigmoid(z)
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = np.mean(p - y)
        w -= lr * gw
        b -= lr * gb
        # 计算交叉熵损失 + L2/2
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) \
            + l2 * np.sum(w * w) / (2 * n)
        if abs(prev_loss - loss) < tol:
            print(f"  第 {it + 1} 步收敛(损失变化 < {tol})")
            break
        prev_loss = loss
    else:
        print(f"  达到最大迭代次数 {n_iter} (未触发早停)")
    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# ==================== 评估指标 ====================
def auc_score(y_true, y_score):
    """AUC (Mann-Whitney U 排序法，O(n log n))"""
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    s = y_score[order]
    n = len(s)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    # 处理并列分数: 赋平均秩
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = (i + 1 + j) / 2.0
        i = j
    rank_full = np.empty(n, dtype=np.float64)
    rank_full[order] = ranks
    sum_ranks_pos = rank_full[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(y_true, y_pred, threshold):
    """给定阈值计算混淆矩阵与指标
    - fp: 误判(实际非违约预测为违约)
    - fn: 漏判(实际违约预测为非违约)
    """
    y_hat = (y_pred >= threshold).astype(int)
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def fmt_itemset(s):
    return "{" + ", ".join(sorted(s)) + "}"


def print_confusion(ev, n_total, tag=""):
    """输出混淆矩阵 + 误判/漏判"""
    print(f"  {'':>16s}  预测非违约  预测违约    合计")
    print(f"  {'实际非违约':>16s}  {ev['tn']:>8d}  {ev['fp']:>8d}  {ev['tn']+ev['fp']:>8d}")
    print(f"  {'实际违约':>16s}  {ev['fn']:>8d}  {ev['tp']:>8d}  {ev['fn']+ev['tp']:>8d}")
    print(f"  {'合计':>16s}  {ev['tn']+ev['fn']:>8d}  {ev['fp']+ev['tp']:>8d}  {n_total:>8d}")
    print(f"\n  {tag}准确率={ev['acc']:.4f}  精确率={ev['prec']:.4f}  "
          f"召回率={ev['rec']:.4f}  F1={ev['f1']:.4f}")
    print(f"  误判(非违约→违约): {ev['fp']} 人, 漏判(违约→非违约): {ev['fn']} 人")


# ==================== 基线模型: 全判不违约 ====================
def baseline_all_no_default(y_valid, y_test=None, threshold=0.5):
    """全判不违约基线: 预测概率恒为 0，阈值 0.5 下全部判为不违约
    - 准确率应等于非违约率，AUC 应为 0.5，精确率分母为 0 时返回 0
    返回 dict:
      p_valid, eval_valid, auc_valid, threshold
      p_test, eval_test, auc_test  (当提供 y_test 时)
    """
    yv = np.asarray(y_valid, dtype=np.float64)
    p_valid = np.zeros_like(yv)
    ev_v = evaluate(yv, p_valid, threshold)
    auc_v = auc_score(yv, p_valid)
    out = {
        "p_valid": p_valid,
        "eval_valid": ev_v,
        "auc_valid": auc_v,
        "threshold": threshold,
    }
    if y_test is not None:
        yt = np.asarray(y_test, dtype=np.float64)
        p_test = np.zeros_like(yt)
        ev_t = evaluate(yt, p_test, threshold)
        auc_t = auc_score(yt, p_test)
        out["p_test"] = p_test
        out["eval_test"] = ev_t
        out["auc_test"] = auc_t
    return out


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="基于 apriori_credit 关联规则的违约逻辑回归模型")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV 文件路径")
    parser.add_argument("--min-support", type=float, default=0.05, help="最小支持度")
    parser.add_argument("--min-confidence", type=float, default=0.30, help="最小置信度")
    parser.add_argument("--min-lift", type=float, default=1.50, help="最小提升度")
    parser.add_argument("--max-antecedent", type=int, default=3, help="前件最大特征数")
    parser.add_argument("--n-features", type=int, default=20, help="选取规则特征数")
    parser.add_argument("--lr", type=float, default=0.02, help="学习率(默认0.02)")
    parser.add_argument("--n-iter", type=int, default=20000, help="最大迭代次数")
    parser.add_argument("--l2", type=float, default=1.0, help="L2 正则化系数")
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 → {args.csv}（已尝试工作目录与脚本目录）")
        sys.exit(1)

    # --- 加载数据 ---
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"].copy()
    df_valid = df[df["split"] == "valid"].copy()
    df_test = df[df["split"] == "test"].copy()
    print(f"  train: {len(df_train)} 条, valid: {len(df_valid)} 条, test: {len(df_test)} 条")

    # --- 离散化(分位数由 train 决定，valid/test 复用) ---
    print("\n离散化...")
    t0 = time.time()
    txs_train, q = discretize(df_train)
    txs_valid, _ = discretize(df_valid, q)
    txs_test, _ = discretize(df_test, q)
    y_train = df_train["default_next_month"].values.astype(np.float64)
    y_valid = df_valid["default_next_month"].values.astype(np.float64)
    y_test = df_test["default_next_month"].values.astype(np.float64)
    print(f"  train 违约率: {y_train.mean():.4f}, valid 违约率: {y_valid.mean():.4f}, test 违约率: {y_test.mean():.4f}")
    print(f"  耗时: {time.time() - t0:.2f} 秒")

    # --- 关联规则挖掘: 复用 apriori_credit.py 的实现 ---
    print("\n使用 apriori_credit.py 的 Apriori 算法挖掘关联规则...")
    t0 = time.time()
    max_k = args.max_antecedent + 1
    freq, total = ap.apriori(txs_train, args.min_support, max_k)
    rules = ap.generate_rules(freq, total, args.min_support, args.min_confidence,
                              args.min_lift, TARGET, args.max_antecedent)
    print(f"  频繁项集: {len(freq)}, 关联规则: {len(rules)}, 耗时: {time.time() - t0:.2f} 秒")

    if not rules:
        print("错误: 未挖掘到关联规则，请放宽阈值")
        sys.exit(1)

    # --- 特征选择 ---
    features = select_features(rules, args.n_features)
    print(f"\n选取 {len(features)} 个非冗余规则特征:")
    for i, f in enumerate(features):
        print(f"  {i + 1:2d}. {fmt_itemset(f['antecedent']):<40s}  "
              f"置信度={f['confidence']:.4f}  提升度={f['lift']:.4f}")

    # --- 构建特征矩阵 ---
    X_train = build_matrix(txs_train, features)
    X_valid = build_matrix(txs_valid, features)
    X_test = build_matrix(txs_test, features)

    # 去除匹配模式完全相同的冗余特征(以 train 为准)
    keep_idx = dedup_features(X_train)
    X_train = X_train[:, keep_idx]
    X_valid = X_valid[:, keep_idx]
    X_test = X_test[:, keep_idx]
    features = [features[i] for i in keep_idx]
    print(f"\n去重后保留 {len(features)} 个特征")

    print(f"\n特征匹配率 (满足前件的比例):")
    for j, f in enumerate(features):
        r_tr = X_train[:, j].mean()
        r_va = X_valid[:, j].mean()
        print(f"  {fmt_itemset(f['antecedent']):<40s}  train={r_tr:.4f}  valid={r_va:.4f}")

    # --- 逻辑回归拟合(仅 train) ---
    print(f"\n拟合逻辑回归 (lr={args.lr}, iter={args.n_iter}, l2={args.l2})...")
    t0 = time.time()
    w, b = logistic_fit(X_train, y_train, args.lr, args.n_iter, args.l2)
    print(f"  耗时: {time.time() - t0:.2f} 秒")

    print("\n模型系数:")
    print(f"  截距 (b): {b:+.4f}")
    for i, f in enumerate(features):
        print(f"  {fmt_itemset(f['antecedent']):<40s}  w = {w[i]:+.4f}")

    # --- 预测概率 ---
    p_train = predict_proba(X_train, w, b)
    p_valid = predict_proba(X_valid, w, b)
    p_test = predict_proba(X_test, w, b)

    # --- AUC ---
    auc_tr = auc_score(y_train, p_train)
    auc_va = auc_score(y_valid, p_valid)
    auc_te = auc_score(y_test, p_test)
    print(f"\nAUC:  train = {auc_tr:.4f}    valid = {auc_va:.4f}    test = {auc_te:.4f}")

    # --- 在 valid 上搜索最佳 F1 阈值 (阈值在验证集上选择) ---
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.05, 0.85, 0.01):
        ev = evaluate(y_valid, p_valid, th)
        if ev["f1"] > best_f1:
            best_f1, best_th = ev["f1"], round(th, 2)
    print(f"\n验证集选定阈值: {best_th} (valid F1={best_f1:.4f})")

    # --- 多阈值评估(train/valid) ---
    thresholds = sorted(set([0.22, 0.30, 0.40, 0.50, best_th]))
    print(f"\n{'=' * 78}")
    print(f"  评估结果 (train 拟合, 阈值由 valid 选定, test 验证)")
    print(f"{'=' * 78}")
    hdr = (f"  {'阈值':>5s} | {'train准确':>8s} {'trainF1':>7s} | "
           f"{'valid准确':>8s} {'valid精确':>8s} {'valid召回':>8s} {'validF1':>7s}  备注")
    print(hdr)
    print("  " + "-" * 74)
    for th in thresholds:
        et = evaluate(y_train, p_train, th)
        ev = evaluate(y_valid, p_valid, th)
        tag = " ← 验证集选定阈值" if abs(th - best_th) < 0.005 else ""
        print(f"  {th:>5.2f} | {et['acc']:>8.4f} {et['f1']:>7.4f} | "
              f"{ev['acc']:>8.4f} {ev['prec']:>8.4f} {ev['rec']:>8.4f} {ev['f1']:>7.4f}{tag}")

    # --- 采用 valid 选定阈值做最终 valid 评估 ---
    th = best_th
    ev = evaluate(y_valid, p_valid, th)
    print(f"\n  混淆矩阵 - valid (阈值 = {th}, 由 valid 选定)")
    print_confusion(ev, len(y_valid), "valid: ")

    # --- 概率分布 ---
    print(f"\n  违约概率分布:")
    print(f"    train: min={p_train.min():.4f}  mean={p_train.mean():.4f}  max={p_train.max():.4f}")
    print(f"    valid: min={p_valid.min():.4f}  mean={p_valid.mean():.4f}  max={p_valid.max():.4f}")
    print(f"    test:  min={p_test.min():.4f}  mean={p_test.mean():.4f}  max={p_test.max():.4f}")
    print(f"  valid 预测违约人数: {int(np.sum(p_valid >= th))} / {len(p_valid)}")

    # --- 违约概率分段统计(等宽 0.1 分箱) ---
    print(f"\n  valid 违约概率分段统计(等宽 0.1):")
    bins = [i / 10.0 for i in range(11)]
    labels = [f"{i/10:.1f}~{(i+1)/10:.1f}" for i in range(10)]
    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            mask = (p_valid >= bins[i]) & (p_valid < bins[i + 1])
        else:
            mask = (p_valid >= bins[i]) & (p_valid <= bins[i + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            actual_rate = y_valid[mask].mean()
            print(f"    {labels[i]:>8s}: {cnt:>5d} 人  实际违约率={actual_rate:.4f}")

    # ==================== 基线模型: 全判不违约 ====================
    print(f"\n{'=' * 78}")
    print(f"  基线模型: 全判不违约 (预测概率恒为 0)")
    print(f"{'=' * 78}")
    baseline = baseline_all_no_default(y_valid, y_test, 0.5)
    for name, ev_b, auc_b, y_true in [("valid", baseline["eval_valid"], baseline["auc_valid"], y_valid),
                                      ("test", baseline["eval_test"], baseline["auc_test"], y_test)]:
        non_default_rate = 1.0 - y_true.mean()
        print(f"\n  [{name}] 样本数: {len(y_true)}    实际违约: {int(y_true.sum())} 人    非违约率: {non_default_rate:.4f}")
        print(f"    准确率={ev_b['acc']:.4f}  精确率={ev_b['prec']:.4f}  召回率={ev_b['rec']:.4f}  F1={ev_b['f1']:.4f}  AUC={auc_b:.4f}")
        print(f"    误判(非违约→违约): {ev_b['fp']} 人, 漏判(违约→非违约): {ev_b['fn']} 人")
        print(f"    校验: 准确率({ev_b['acc']:.4f}) == 非违约率({non_default_rate:.4f}): "
              f"{'✓' if abs(ev_b['acc'] - non_default_rate) < 1e-9 else '✗'}")

    # ==================== test 集最终评价(锁定模型与阈值) ====================
    print(f"\n{'=' * 78}")
    print(f"  test 集最终评价 (模型与阈值已锁定, 仅做一次性评价)")
    print(f"{'=' * 78}")
    print(f"  样本数: {len(y_test)}    实际违约: {int(y_test.sum())} 人    实际违约率: {y_test.mean():.4f}")
    print(f"  ROC-AUC: {auc_te:.4f}")
    ev_te = evaluate(y_test, p_test, th)
    print(f"\n  混淆矩阵 - test (阈值 = {th})")
    print_confusion(ev_te, len(y_test), "test: ")
    print(f"  test 预测违约人数: {int(np.sum(p_test >= th))} / {len(p_test)}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()