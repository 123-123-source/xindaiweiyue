import joblib
import numpy as np
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. 同学A的纯手写算法
# ==========================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_fit(X, y, lr=0.02, n_iter=20000, l2=1.0, tol=1e-7):
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    prev_loss = float("inf")
    for it in range(n_iter):
        z = X @ w + b
        p = sigmoid(z)
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = np.mean(p - y)
        w -= lr * gw
        b -= lr * gb

        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + l2 * np.sum(w * w) / (2 * n)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return w, b


def evaluate(y_true, y_pred):
    """计算准确率、召回率、F1 和混淆矩阵四值"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ==========================================
# 2. 可导入的训练 / 预测接口（供 app.py 复用）
# ==========================================
def train_logistic_2(X_train, y_train, lr=0.02, n_iter=20000, l2=1.0, tol=1e-7):
    """用同学A的梯度下降 L2 逻辑回归在训练集上拟合，返回 {w, b}"""
    y_train = np.asarray(y_train).astype(np.float64)
    w, b = logistic_fit(X_train, y_train, lr, n_iter, l2, tol)
    return {"w": w, "b": b}


def predict_proba(X, w, b):
    """给定训练好的 w, b，输出违约概率"""
    return sigmoid(np.asarray(X) @ w + b)


def select_threshold(p_valid, y_valid):
    """在验证集上按 F1 自动选阈值，返回 (best_th, best_f1)"""
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.05, 0.85, 0.01):
        ev = evaluate(y_valid, (p_valid >= th).astype(int))
        if ev["f1"] > best_f1:
            best_f1, best_th = ev["f1"], round(th, 2)
    return best_th, best_f1


def load_prepared_data(path="prepared_data.joblib"):
    """加载 prepared_data.joblib，按行数 20999/4500/4501 自动识别 6 个数组"""
    data = joblib.load(path)
    X_train = X_valid = X_test = None
    y_train = y_valid = y_test = None
    for item in data:
        arr = np.array(item)
        if len(arr.shape) == 1 or (len(arr.shape) == 2 and arr.shape[1] == 1):
            n = arr.shape[0]
            if n == 20999:
                y_train = arr.ravel().astype(np.float64)
            elif n == 4500:
                y_valid = arr.ravel().astype(np.float64)
            else:
                y_test = arr.ravel().astype(np.float64)
        elif len(arr.shape) == 2 and arr.shape[1] == 90:
            n = arr.shape[0]
            if n == 20999:
                X_train = arr.astype(np.float64)
            elif n == 4500:
                X_valid = arr.astype(np.float64)
            else:
                X_test = arr.astype(np.float64)
    if any(v is None for v in [X_train, X_valid, X_test, y_train, y_valid, y_test]):
        raise RuntimeError("prepared_data.joblib 自动识别失败")
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ==========================================
# 3. 脚本主入口：完整训练 + 评估流程
# ==========================================
def _print_result(name, ev, y_true, y_prob, best_th):
    print(f"\n{'='*40}")
    print(f"=== {name} 结果 (阈值={best_th}) ===")
    print(f"{'='*40}")
    print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"准确率  : {ev['acc']:.4f}")
    print(f"精确率  : {ev['prec']:.4f}")
    print(f"召回率  : {ev['rec']:.4f}")
    print(f"F1 值   : {ev['f1']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"              预测非违约  预测违约    合计")
    print(f"  实际非违约    {ev['tn']:>7d}    {ev['fp']:>7d}  {ev['tn']+ev['fp']:>7d}")
    print(f"  实际违约      {ev['fn']:>7d}    {ev['tp']:>7d}  {ev['fn']+ev['tp']:>7d}")
    print(f"  合计          {ev['tn']+ev['fn']:>7d}    {ev['fp']+ev['tp']:>7d}  {ev['tn']+ev['fp']+ev['fn']+ev['tp']:>7d}")


if __name__ == "__main__":
    print("正在加载你的 prepared_data.joblib...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_prepared_data()

    print("\n" + "="*40)
    print("🔒 数据防泄露声明：")
    print(f"1. 训练阶段，模型只会看到：X_train {X_train.shape} 和 y_train {y_train.shape}")
    print(f"2. 验证集(选阈值)：X_valid {X_valid.shape} 和 y_valid {y_valid.shape}")
    print(f"3. 测试集(一次性评估)：X_test {X_test.shape} 和 y_test {y_test.shape}")
    print("="*40 + "\n")

    print("开始用同学A的算法训练你的数据... (正在锻炼大脑...)")
    model = train_logistic_2(X_train, y_train, lr=0.02, n_iter=20000)
    w, b = model["w"], model["b"]

    p_valid = predict_proba(X_valid, w, b)
    p_test = predict_proba(X_test, w, b)

    best_th, best_f1 = select_threshold(p_valid, y_valid)
    print(f"\n验证集自动选定阈值: {best_th} (valid F1={best_f1:.4f})")

    y_pred_valid = (p_valid >= best_th).astype(int)
    y_pred_test = (p_test >= best_th).astype(int)
    ev_valid = evaluate(y_valid, y_pred_valid)
    ev_test = evaluate(y_test, y_pred_test)

    _print_result("验证集 (Valid)", ev_valid, y_valid, p_valid, best_th)
    _print_result("测试集 (Test)", ev_test, y_test, p_test, best_th)

    print("\n" + "="*40)
    print("✅ 实验完成，成绩单已出！")