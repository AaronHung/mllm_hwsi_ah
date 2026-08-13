"""
路線 A 定量分析(A.2)共用工具函式。

方法說明:
- proxy A-distance: 用簡單 logistic regression 當 domain classifier 區分 site,
  A-distance = 4*acc - 2(acc=0.5 時為 0,acc=1.0 時為 2),acc 用重複分層交叉驗證的平均值。
- MMD: RBF kernel、median heuristic 決定 bandwidth 的 unbiased MMD^2 估計。
- 所有分類器都是簡單模型(logistic regression, L2 正則),嚴禁深層網路(依 CLAUDE.md 規定)。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _fit_predict_acc(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(X_train), y_train)
    pred = clf.predict(scaler.transform(X_test))
    return (pred == y_test).mean()


def proxy_a_distance_wsi_level(X, y, n_splits=3, n_repeats=30, random_state=0):
    """WSI 層級:每個樣本是一張 WSI(mean-pooled 向量),無需 group,直接分層 K-fold。"""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    repeat_accs = []
    cur_repeat_fold_accs = []
    last_repeat_idx = -1
    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_idx = i // n_splits
        if repeat_idx != last_repeat_idx and cur_repeat_fold_accs:
            repeat_accs.append(np.mean(cur_repeat_fold_accs))
            cur_repeat_fold_accs = []
        last_repeat_idx = repeat_idx
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        acc = _fit_predict_acc(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        cur_repeat_fold_accs.append(acc)
    if cur_repeat_fold_accs:
        repeat_accs.append(np.mean(cur_repeat_fold_accs))
    repeat_accs = np.array(repeat_accs)
    a_dist = 4 * repeat_accs - 2
    return {
        "acc_mean": float(repeat_accs.mean()),
        "acc_std": float(repeat_accs.std()),
        "a_distance_mean": float(a_dist.mean()),
        "a_distance_std": float(a_dist.std()),
        "n_repeats": len(repeat_accs),
    }


def proxy_a_distance_instance_level(X, y, groups, n_splits=3, n_repeats=30, random_state=0):
    """Instance 層級:同一張 WSI 的所有 instance(region)必須同進同出,避免資訊洩漏。

    當某組 site 只有極少數 WSI(例如 n=3)時,StratifiedGroupKFold 偶爾會把整組
    WSI 全分進同一個 fold,導致訓練集裡只剩一個類別——這種切分直接跳過、不計入平均。
    """
    repeat_accs = []
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep)
        fold_accs = []
        for train_idx, test_idx in sgkf.split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            acc = _fit_predict_acc(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
            fold_accs.append(acc)
        if fold_accs:
            repeat_accs.append(np.mean(fold_accs))
    repeat_accs = np.array(repeat_accs)
    a_dist = 4 * repeat_accs - 2
    return {
        "acc_mean": float(repeat_accs.mean()),
        "acc_std": float(repeat_accs.std()),
        "a_distance_mean": float(a_dist.mean()),
        "a_distance_std": float(a_dist.std()),
        "n_repeats": len(repeat_accs),
    }


def mmd_rbf(X, Y, gamma=None):
    """Unbiased MMD^2 estimate,RBF kernel,預設用 median heuristic 決定 gamma。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    scaler = StandardScaler().fit(np.vstack([X, Y]))
    X = scaler.transform(X)
    Y = scaler.transform(Y)

    if gamma is None:
        Z = np.vstack([X, Y])
        n = Z.shape[0]
        if n > 1500:
            rng = np.random.RandomState(0)
            idx = rng.choice(n, 1500, replace=False)
            Z = Z[idx]
        sq_dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        med = np.median(sq_dists[sq_dists > 0])
        gamma = 1.0 / (med + 1e-12)

    def rbf(A, B):
        sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * sq)

    m, n = X.shape[0], Y.shape[0]
    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.sum() / (m * n)

    mmd2 = sum_xx + sum_yy - 2 * sum_xy
    return float(mmd2)
