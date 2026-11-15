import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
import hashlib
import pathlib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import logistic_credit_v2 as lcv2
import logistic_credit_v2_new as lcv2_new
import logistic_2 as l2mod

st.set_page_config(page_title="风险分析系统", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "SimHei", "Segoe UI", sans-serif;
}
body {
    background:
        radial-gradient(ellipse 80% 50% at 50% 0%, rgba(20, 40, 40, 0.4) 0%, transparent 60%),
        radial-gradient(circle at 1px 1px, rgba(30, 50, 50, 0.12) 1px, transparent 0),
        linear-gradient(160deg, #0f1c1c 0%, #0a0f14 40%, #060a0e 100%);
    background-size: 100% 100%, 48px 48px, 100% 100%;
    background-attachment: fixed;
}
.stApp {
    background: transparent !important;
}
div[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
}
#bg-particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    pointer-events: none;
}
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
footer:after {
    content: "© 客户风险分析系统 · 教学示例";
    visibility: visible;
    display: block;
    text-align: center;
    color: #94a3b8;
    padding: 8px;
    font-size: 0.85rem;
}
p, li, label, td, code, pre {
    color: #cbd5e1 !important;
}
h1 {
    background: linear-gradient(90deg, #5eead4 0%, #ffffff 60%, #a7f3d0 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: 1px;
}
h2 {
    border-left: 5px solid #5eead4;
    padding-left: 12px !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
h3 {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,50,50,0.55) 0%, rgba(15,28,28,0.65) 100%);
    border: 1px solid rgba(94,234,212,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(94,234,212,0.18);
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}
div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] {
    background: rgba(6,10,14,0.55) !important;
    border-right: 2px solid #5eead4;
    backdrop-filter: blur(1px);
    -webkit-backdrop-filter: blur(1px);
}
section[data-testid="stSidebar"] > div,
div[data-testid="stSidebarContentContainer"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    -webkit-text-fill-color: #e2e8f0;
    color: #e2e8f0 !important;
    background: none;
    border-left: none;
    padding-left: 0;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div[role="radio"] {
    color: #cbd5e1 !important;
}
.stButton > button {
    border-radius: 8px !important;
    background: linear-gradient(90deg, #5eead4, #38bdf8) !important;
    color: #0f1c1c !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 8px 22px !important;
    box-shadow: 0 2px 8px rgba(94,234,212,0.25) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(94,234,212,0.4) !important;
}
table {
    border-radius: 8px !important;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
thead tr th {
    background: #0f1c1c !important;
    color: #5eead4 !important;
    font-weight: 600 !important;
}
tbody tr {
    background: rgba(15,28,28,0.6) !important;
    color: #e2e8f0 !important;
}
tbody tr:nth-child(even) {
    background: rgba(20,40,40,0.6) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border-color: rgba(94,234,212,0.3) !important;
}
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #5eead4, transparent);
    margin: 16px 0;
}
/* 数字卡片：根据页面宽度自动调整每行个数（宽屏一行多个，窄屏自动换行成多行） */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) {
    flex-wrap: wrap !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) > div {
    flex: 1 1 170px !important;
    min-width: 170px !important;
}
/* 回退规则：不支持 :has() 的浏览器环境，对所有横向块启用换行与最小宽度 */
@supports not selector(:has(*)) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div {
        flex: 1 1 170px !important;
        min-width: 170px !important;
    }
}
/* 数字卡片的数值完整显示，不省略截断 */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="stMetricLabel"] {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
</style>
""",
    unsafe_allow_html=True,
)

import streamlit.components.v1 as components
components.html(
    """
<script>
(function(){
  var pwin = parent;
  var doc = pwin.document;
  var canvas = doc.getElementById('bg-particles');
  if (!canvas) {
    canvas = doc.createElement('canvas');
    canvas.id = 'bg-particles';
    doc.body.appendChild(canvas);
  }
  if (pwin.__bgParticlesRaf) pwin.cancelAnimationFrame(pwin.__bgParticlesRaf);
  var ctx = canvas.getContext('2d');
  var w, h, particles;
  function resize(){ w = canvas.width = pwin.innerWidth; h = canvas.height = pwin.innerHeight; }
  function init(){
    particles = [];
    var count = Math.min(140, Math.floor(w*h/11000));
    for(var i=0;i<count;i++){
      particles.push({x:Math.random()*w, y:Math.random()*h, vx:(Math.random()-0.5)*0.5, vy:(Math.random()-0.5)*0.5, r:(Math.random()*1.8+0.6)*2});
    }
  }
  function draw(){
    ctx.clearRect(0,0,w,h);
    for(var i=0;i<particles.length;i++){
      var p=particles[i];
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>w) p.vx*=-1;
      if(p.y<0||p.y>h) p.vy*=-1;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle='rgba(94,234,212,0.85)'; ctx.fill();
    }
    for(var i=0;i<particles.length;i++){
      for(var j=i+1;j<particles.length;j++){
        var dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
        var d=Math.sqrt(dx*dx+dy*dy);
        if(d<260){
          ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y); ctx.lineTo(particles[j].x,particles[j].y);
          ctx.strokeStyle='rgba(94,234,212,'+(0.2*(1-d/260))+')'; ctx.stroke();
        }
      }
    }
    pwin.__bgParticlesRaf = pwin.requestAnimationFrame(draw);
  }
  pwin.addEventListener('resize', function(){ resize(); init(); });
  resize(); init(); draw();
})();
</script>
""",
    height=0,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "#0a0f14",
    "axes.facecolor": "#0f1c1c",
    "axes.edgecolor": "#5eead4",
    "axes.labelcolor": "#cbd5e1",
    "axes.titlecolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#cbd5e1",
    "grid.color": "#1a2a2a",
    "grid.alpha": 0.4,
    "axes.grid": True,
    "legend.facecolor": "#0f1c1c",
    "legend.edgecolor": "#5eead4",
    "legend.labelcolor": "#cbd5e1",
})

DATA_FILE = "01_credit_default.csv"
PREPARED_DATA_JOBLIB = "prepared_data.joblib"
CACHE_DIR = "__model_cache__"
BUSINESS_RULES_FILE = "business_rules.json"
_RULES_REQUIRED_FIELDS = {"模型名称", "模型文件", "阈值", "说明", "概率规则", "特征规则", "核心特征方向", "分层决策逻辑", "最优组合指标"}
MODEL_TO_RULES_FILE = {
    "random_forest": "business_rules.json",
    "random_forest_calibrated": "business_rules.json",
    "logistic_90dim": "business_rules_lr2.json",
    "logistic": "business_rules_lcv2.json",
}
_RULES_REQUIRED_FIELDS_RF = {"模型名称", "模型文件", "阈值", "说明", "概率规则", "特征规则", "核心特征方向", "分层决策逻辑", "最优组合指标"}
_RULES_REQUIRED_FIELDS_LR2 = {"模型名称", "模型文件", "阈值", "说明", "概率规则", "特征规则", "核心特征方向", "分层决策逻辑", "最优组合指标", "跨模型交叉验证"}
_RULES_REQUIRED_FIELDS_LCV2 = {"模型名称", "模型文件", "阈值", "说明", "概率规则", "特征规则", "分层决策逻辑", "规则生成方式"}
RULES_PAGE_MODEL_OPTIONS = [("random_forest", "随机森林"), ("logistic_90dim", "逻辑回归2"), ("logistic", "逻辑回归(关联规则)")]
_FILE_TO_FIELDS = {
    "business_rules.json": _RULES_REQUIRED_FIELDS_RF,
    "business_rules_lr2.json": _RULES_REQUIRED_FIELDS_LR2,
    "business_rules_lcv2.json": _RULES_REQUIRED_FIELDS_LCV2,
}


def get_rules_file_for_model(locked_model: str) -> str | None:
    return MODEL_TO_RULES_FILE.get(locked_model)


def get_file_signature(path: str) -> str:
    try:
        return f"{os.path.getmtime(path):.6f}_{os.path.getsize(path)}"
    except OSError:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:8]


def compute_data_sig() -> str:
    return get_file_signature(DATA_FILE)


def compute_model_sig() -> str:
    parts = [get_file_signature(DATA_FILE),
             get_file_signature(RF_JOBLIB),
             get_file_signature(RF_CAL_JOBLIB),
             get_file_signature(PREPARED_DATA_JOBLIB)]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def build_cache_filename(data_sig: str) -> str:
    return f"model_bundle_{data_sig[:8]}.joblib"


def get_cache_path() -> pathlib.Path:
    cache_dir = pathlib.Path(__file__).parent / CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir / build_cache_filename(compute_data_sig())


def load_disk_cache(path: pathlib.Path, expected_sig: str):
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and bundle.get("_cache_sig") == expected_sig:
            bundle.pop("_cache_sig", None)
            return bundle
        return None
    except Exception as e:
        print(f"[cache] 磁盘缓存读取失败 {path}: {type(e).__name__}: {e}")
        return None


def save_disk_cache(path: pathlib.Path, bundle: dict, sig: str) -> bool:
    try:
        bundle_copy = dict(bundle)
        bundle_copy["_cache_sig"] = sig
        tmp_path = path.with_suffix(".tmp")
        joblib.dump(bundle_copy, tmp_path, compress=3)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"[cache] 磁盘缓存写入失败 {path}: {type(e).__name__}: {e}")
        return False

PAY_COLS = [
    "pay_status_2005_09",
    "pay_status_2005_08",
    "pay_status_2005_07",
    "pay_status_2005_06",
    "pay_status_2005_05",
    "pay_status_2005_04",
]

MODEL_LABELS = {
    "logistic": "逻辑回归",
    "logistic_90dim": "逻辑回归2",
    "random_forest": "随机森林",
    "random_forest_calibrated": "随机森林(校准)",
    "baseline": "全判不违约基线",
}

K_SEGMENTS = [
    {"lower": 0.0, "upper": 1.31, "model_key": "logistic"},
    {"lower": 1.31, "upper": 3.97, "model_key": "logistic_90dim"},
    {"lower": 3.97, "upper": 10.73, "model_key": "random_forest_calibrated"},
    {"lower": 10.73, "upper": float("inf"), "model_key": "random_forest"},
]


def recommend_model_by_k(k: float) -> dict:
    if k < 0:
        k = 0.0
    for seg in K_SEGMENTS:
        if seg["lower"] <= k < seg["upper"]:
            return {"model_key": seg["model_key"], "lower": seg["lower"], "upper": seg["upper"]}
    last = K_SEGMENTS[-1]
    return {"model_key": last["model_key"], "lower": last["lower"], "upper": last["upper"]}


LOCKED_STATE_FILE = "locked_state.json"

DEFAULT_MODEL_THRESHOLDS = {
    "random_forest": 0.14,
    "random_forest_calibrated": 0.14,
    "logistic": 0.17,
    "logistic_90dim": 0.26,
}


def get_locked_threshold(model_key: str) -> float:
    mts = st.session_state.get("model_thresholds", {})
    if model_key in mts:
        return float(mts[model_key])
    if model_key in DEFAULT_MODEL_THRESHOLDS:
        return float(DEFAULT_MODEL_THRESHOLDS[model_key])
    return 0.5


def is_model_locked(model_key: str) -> bool:
    return model_key in (st.session_state.get("model_thresholds", {}) or {})


def load_locked_state_to_session():
    if not os.path.exists(LOCKED_STATE_FILE):
        return
    try:
        with open(LOCKED_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return
    if not isinstance(state, dict):
        return
    mts = state.get("model_thresholds", {})
    if not isinstance(mts, dict):
        mts = {}
    if not mts and state.get("locked") and "locked_model" in state and "locked_threshold" in state:
        mts = {state["locked_model"]: float(state["locked_threshold"])}
    if mts:
        existing = dict(st.session_state.get("model_thresholds", {}) or {})
        for k, v in mts.items():
            existing.setdefault(k, float(v))
        st.session_state.setdefault("model_thresholds", existing)
        st.session_state.setdefault("locked", True)


def save_locked_state():
    state = {
        "locked": bool(st.session_state.get("locked", False)),
        "model_thresholds": st.session_state.get("model_thresholds", {}) or {},
    }
    try:
        with open(LOCKED_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_recommend_message(k: float, current_model_key: str) -> tuple:
    rec = recommend_model_by_k(k)
    rec_model_key = rec["model_key"]
    lower, upper = rec["lower"], rec["upper"]
    rec_name = MODEL_LABELS.get(rec_model_key, rec_model_key)
    if upper == float("inf"):
        interval_str = f"[{lower}, +∞)"
    else:
        interval_str = f"[{lower}, {upper})"
    k_str = f"{k:g}"
    if rec_model_key == current_model_key:
        hint_type = "confirm"
        message = f"当前 k={k_str}（区间 {interval_str}），所选「{rec_name}」总成本最低，为最优模型。"
    else:
        hint_type = "guide"
        cur_name = MODEL_LABELS.get(current_model_key, current_model_key)
        message = f"当前 k={k_str}（区间 {interval_str}），推荐模型为「{rec_name}」（总成本最低）。当前所选「{cur_name}」非最优，建议切换。"
    return message, hint_type


def render_recommend_hint(k: float, model_key: str) -> None:
    try:
        message, hint_type = build_recommend_message(k, model_key)
        if hint_type == "confirm":
            st.success(message)
        else:
            st.info(message)
    except Exception:
        st.warning("推荐提示渲染失败")


def validate_rules_fields(rules: dict) -> bool:
    if not isinstance(rules, dict):
        return False
    return _RULES_REQUIRED_FIELDS.issubset(rules.keys())


def validate_rules_fields_by_type(rules: dict, file_path: str) -> bool:
    if not isinstance(rules, dict):
        return False
    required = _FILE_TO_FIELDS.get(file_path, _RULES_REQUIRED_FIELDS)
    return required.issubset(rules.keys())


@st.cache_data
def load_business_rules(_rules_sig: str = None, file_path: str = BUSINESS_RULES_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        if not validate_rules_fields_by_type(rules, file_path):
            return None, "format_error"
        return rules, "ok"
    except FileNotFoundError:
        return None, "missing"
    except (json.JSONDecodeError, OSError):
        return None, "format_error"


def compute_overdue_flag(feature_values: dict, feature_cols: list) -> tuple:
    total = 0
    missing_flags = []
    for col in feature_cols:
        v = feature_values.get(col, 0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0
            missing_flags.append(col)
        total += int(v)
    return total, missing_flags


def apply_rules_adapt(prob, features, feature_names, rules, locked_threshold=None) -> dict:
    try:
        from apply_rules import apply_rules as _apply
        result = _apply(prob, features, feature_names, rules, locked_threshold)
        result["降级"] = False
        result["降级原因"] = None
        return result
    except ImportError:
        return {"决策": None, "原因": "", "概率": prob, "逾期标志": 0, "降级": True, "降级原因": "模块加载失败"}
    except (TypeError, KeyError, Exception) as e:
        return {"决策": None, "原因": "", "概率": prob, "逾期标志": 0, "降级": True, "降级原因": f"执行失败: {type(e).__name__}: {e}"}


def rules_result_to_steps_and_final(result: dict, rules: dict) -> tuple:
    decision_map = {"通过": ("green", False), "灰名单": ("yellow", True), "拒绝": ("red", False)}
    color, review = decision_map.get(result.get("决策"), ("gray", False))
    steps = []
    for i, s in enumerate(rules.get("分层决策逻辑", []), 1):
        is_last = (i == len(rules.get("分层决策逻辑", [])))
        steps.append({
            "step": s["步骤"],
            "说明": s["说明"],
            "input": result.get("原因", "") if is_last else "",
            "conclusion": result["决策"] if is_last else "执行中",
            "terminated": is_last,
        })
    final = {
        "结论档位": result.get("决策"),
        "颜色": color,
        "触发规则": result.get("原因", ""),
        "触发步骤": len(steps),
        "人工审核": review,
    }
    return steps, final


def execute_layered_decision(prob, overdue_flag, rules, locked_threshold=None) -> tuple:
    """
    分层决策执行器：基于概率、逾期标志与规则，按分层决策逻辑得出最终档位。
    与 apply_rules.apply_rules 保持同一决策顺序，但直接基于 overdue_flag
    （由 page_demo 预先计算），避免对完整特征向量的依赖。
    返回: (steps, final_decision)，结构同 rules_result_to_steps_and_final。
    """
    threshold = rules["阈值"]
    if not isinstance(threshold, (int, float)):
        threshold = locked_threshold if locked_threshold is not None else 0.5

    feat_rules = rules.get("特征规则", {})
    has_overdue = "逾期标志" in feat_rules
    has_top_assoc = "Top关联规则" in feat_rules

    decision = None
    reason = None

    if has_overdue:
        overdue_rules = feat_rules["逾期标志"]["规则"]
        has_hard_rule = any(">= 2" in r["条件"] for r in overdue_rules)
        if has_hard_rule and overdue_flag >= 2:
            decision = "拒绝"
            reason = f"硬规则: 逾期标志={overdue_flag}>=2, 多月逾期直接拒绝"
        elif overdue_flag == 1:
            decision = "灰名单"
            reason = f"灰名单: 逾期标志=1, 需人工审核"

    if decision is None and has_top_assoc:
        for tr in feat_rules["Top关联规则"]:
            lift = tr.get("提升度", 0)
            if lift > 2.8:
                decision = "灰名单"
                reason = f"灰名单: 命中高提升度规则(lift={lift}), 需人工审核"
                break

    if decision is None:
        if prob < threshold:
            decision = "通过"
            reason = f"自动通过: 概率{prob:.4f} < 阈值{threshold}"
        elif prob < 0.30:
            decision = "灰名单"
            reason = f"灰名单: 概率{prob:.4f} 在 [{threshold}, 0.30) 区间, 需人工审核"
        else:
            decision = "拒绝"
            reason = f"拒绝: 概率{prob:.4f} >= 0.30, 违约风险过高"

    result = {"决策": decision, "原因": reason, "概率": round(float(prob), 4), "逾期标志": overdue_flag}
    return rules_result_to_steps_and_final(result, rules)


def render_decision_steps(steps: list) -> None:
    for s in steps:
        icon = "🔹" if not s["terminated"] else "⏹️"
        with st.expander(f"{icon} 步骤 {s['step']}：{s['说明']}", expanded=True):
            st.markdown(f"**输入**：{s['input']}")
            st.markdown(f"**结论**：{s['conclusion']}")


def render_final_decision_card(final_decision: dict) -> None:
    if final_decision is None or final_decision["结论档位"] is None:
        st.warning("决策结论异常")
        return
    color_map = {"green": ("✅", "#34d399"), "yellow": ("⚠️", "#fbbf24"), "red": ("❌", "#f472b6"), "gray": ("❓", "#9ca3af")}
    icon, color = color_map.get(final_decision["颜色"], ("❓", "#9ca3af"))
    review_tag = "（需人工审核）" if final_decision["人工审核"] else ""
    st.markdown(
        f'<div style="border:2px solid {color};border-radius:12px;padding:16px;margin:8px 0;background:rgba(0,0,0,0.2);">'
        f'<span style="font-size:28px">{icon}</span> '
        f'<span style="font-size:22px;color:{color};font-weight:bold">{final_decision["结论档位"]}{review_tag}</span><br>'
        f'<span style="font-size:14px;color:#aaa">触发规则：{final_decision["触发规则"]}　|　触发步骤：{final_decision["触发步骤"]}</span>'
        f'</div>', unsafe_allow_html=True
    )


def render_probability_rules(rules: dict, current_prob) -> None:
    prob_rules = rules["概率规则"]
    threshold = rules["阈值"]
    is_auto = not isinstance(threshold, (int, float))
    rows = []
    for key in ["通过", "灰名单", "拒绝"]:
        r = prob_rules[key]
        highlight = ""
        if not is_auto and current_prob is not None and not (isinstance(current_prob, float) and np.isnan(current_prob)) and 0 <= current_prob <= 1:
            try:
                if key == "通过" and current_prob < threshold:
                    highlight = " ← 当前"
                elif key == "灰名单" and threshold <= current_prob < 0.30:
                    highlight = " ← 当前"
                elif key == "拒绝" and current_prob >= 0.30:
                    highlight = " ← 当前"
            except TypeError:
                pass
        rows.append({"档位": key + highlight, "条件": r["条件"], "动作": r["动作"], "颜色": r["颜色"]})
    if is_auto:
        st.caption("阈值自动选定，不高亮当前档位")
    st.table(pd.DataFrame(rows))


def render_feature_rules(rules: dict, current_overdue_flag) -> None:
    if "逾期标志" not in rules.get("特征规则", {}):
        return
    fr = rules["特征规则"]["逾期标志"]
    st.markdown(f"**特征列**：{', '.join(fr['特征列'])}")
    st.markdown(f"**说明**：{fr['说明']}")
    st.markdown(f"**合并方式**：{fr['合并方式']}")
    rows = []
    for r in fr["规则"]:
        highlight = ""
        if current_overdue_flag is not None and isinstance(current_overdue_flag, int) and current_overdue_flag >= 0:
            cond = r["条件"]
            if cond == "逾期标志 == 0" and current_overdue_flag == 0:
                highlight = " ← 当前"
            elif cond == "逾期标志 == 1" and current_overdue_flag == 1:
                highlight = " ← 当前"
            elif cond == "逾期标志 >= 2" and current_overdue_flag >= 2:
                highlight = " ← 当前"
        rows.append({"条件": r["条件"] + highlight, "动作": r["动作"], "影响": r["影响"]})
    st.table(pd.DataFrame(rows))


def render_top_association_rules(rules: dict, current_features: dict = None) -> None:
    try:
        top_rules = rules["特征规则"]["Top关联规则"]
        rows = []
        for tr in top_rules:
            rows.append({
                "规则": tr["规则"], "置信度": tr["置信度"], "提升度": tr["提升度"],
                "支持度": tr["支持度"], "解读": tr["解读"],
            })
        st.table(pd.DataFrame(rows))
    except KeyError:
        st.warning("Top关联规则展示失败")


def render_core_feature_directions(rules: dict, feature_values: dict) -> None:
    directions = rules.get("核心特征方向", {})
    rows = []
    has_values = bool(feature_values)
    for feat, info in directions.items():
        row = {"特征": feat, "含义": info["含义"]}
        if "系数" in info:
            row["系数"] = info["系数"]
            row["方向"] = info["方向"]
            row["策略"] = info["策略"]
        else:
            row["方向"] = info["方向"]
            row["类型"] = info.get("类型", "")
            row["策略"] = info["策略"]
        if "↑" not in row["方向"] and "↓" not in row["方向"]:
            if "正" in info["方向"] or "值越大" in info["方向"]:
                row["方向"] = "↑ " + info["方向"]
            elif "负" in info["方向"] or "值越小" in info["方向"]:
                row["方向"] = "↓ " + info["方向"]
        if has_values:
            v = feature_values.get(feat)
            row["当前值"] = "缺失" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append(row)
    st.table(pd.DataFrame(rows))


def render_optional_fields(rules: dict) -> None:
    if "最优组合指标" in rules:
        st.markdown("**最优组合指标**")
        opt = rules["最优组合指标"]
        for k, v in opt.items():
            st.markdown(f"- {k}：{v}")
    if "跨模型交叉验证" in rules:
        st.markdown("**跨模型交叉验证**")
        cv = rules["跨模型交叉验证"]
        rows = []
        for feat, info in cv.items():
            rows.append({"特征": feat, "随机森林SHAP": info.get("随机森林SHAP", ""),
                         "逻辑回归系数": info.get("逻辑回归系数", ""), "一致": info.get("一致", "")})
        st.table(pd.DataFrame(rows))
    if "规则生成方式" in rules:
        st.markdown("**规则生成方式**")
        gen = rules["规则生成方式"]
        for k, v in gen.items():
            st.markdown(f"- {k}：{v}")



RF_JOBLIB = "best_risk_model.joblib"
RF_CAL_JOBLIB = "best_risk_model_calibrated.joblib"


def make_sample(f31, f0, f29, f42, f8, f64, f53, f74):
    x = np.zeros(90)
    x[31] = f31
    x[0] = f0
    x[29] = f29
    x[42] = f42
    x[8] = f8
    x[64] = f64
    x[53] = f53
    x[74] = f74
    return x


@st.cache_data
def load_data(_data_sig: str = None):
    return pd.read_csv(DATA_FILE)


PREPROCESSED_BUNDLE = "preprocessed_bundle.joblib"

_BUNDLE_REQUIRED_FIELDS = {
    "q", "features", "keep_idx", "X_train", "X_valid", "X_test",
    "w", "b", "w90", "b90", "p_valid_apriori", "p_valid_90dim",
    "y_valid", "y_test", "Xtr_ready", "Xva_ready", "Xte_ready", "version",
}


def load_preprocessed_bundle():
    path = pathlib.Path(__file__).parent / PREPROCESSED_BUNDLE
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        missing = _BUNDLE_REQUIRED_FIELDS - set(bundle.keys())
        if missing:
            print(f"[preprocess] 已处理文件字段缺失: {missing}")
            return None
        return bundle
    except Exception as e:
        print(f"[preprocess] 已处理文件损坏: {type(e).__name__}: {e}")
        return None


def verify_bundle_version(bundle: dict) -> bool:
    ver = bundle.get("version", {})
    csv_sig = get_file_signature(DATA_FILE)
    prepared_data_sig = get_file_signature(PREPARED_DATA_JOBLIB)
    return ver.get("csv_sig") == csv_sig and ver.get("prepared_data_sig") == prepared_data_sig


@st.cache_resource
def train_models(_model_sig: str = None):
    preprocessed = load_preprocessed_bundle()
    if preprocessed is None:
        st.error("未找到已处理文件 preprocessed_bundle.joblib，请先运行预处理脚本：python preprocess_offline.py")
        st.stop()
    if not verify_bundle_version(preprocessed):
        st.error("已处理文件版本与当前数据不匹配，请重新运行预处理脚本：python preprocess_offline.py")
        st.stop()
    if preprocessed["version"].get("consistency_passed") is False:
        st.error("已处理文件一致性自检未通过，请重新运行预处理脚本：python preprocess_offline.py")
        st.stop()

    rf = joblib.load(RF_JOBLIB)
    rf_cal = joblib.load(RF_CAL_JOBLIB)

    Xva_ready = preprocessed["Xva_ready"]
    y_valid = preprocessed["y_valid"]

    model_new = {
        "w": preprocessed["w"],
        "b": preprocessed["b"],
        "features": preprocessed["features"],
        "q": preprocessed["q"],
        "keep_idx": preprocessed["keep_idx"],
        "p_valid": preprocessed["p_valid_apriori"],
    }
    lg2 = {"w": preprocessed["w90"], "b": preprocessed["b90"]}

    valid_proba = {
        "logistic": preprocessed["p_valid_apriori"],
        "logistic_90dim": preprocessed["p_valid_90dim"],
        "random_forest": rf.predict_proba(Xva_ready)[:, 1],
        "random_forest_calibrated": rf_cal.predict_proba(Xva_ready)[:, 1],
        "baseline": lcv2.baseline_all_no_default(y_valid)["p_valid"],
    }

    data = load_data(compute_data_sig())
    feature_cols = [c for c in data.columns if c not in ("record_id", "split", "default_next_month")]

    return {
        "models": {
            "logistic": model_new,
            "logistic_90dim": lg2,
            "random_forest": rf,
            "random_forest_calibrated": rf_cal,
            "baseline": None,
        },
        "feature_cols": feature_cols,
        "valid_proba": valid_proba,
        "valid_y": y_valid,
        "X_ready": {
            "train": preprocessed["Xtr_ready"],
            "valid": preprocessed["Xva_ready"],
            "test": preprocessed["Xte_ready"],
        },
        "y_test": preprocessed["y_test"],
    }


def compute_metrics(proba, y, threshold):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    res = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "false_positive": fp, "false_negative": fn}
    if tp + fp == 0:
        res["precision"] = 0.0
        res["precision_reason"] = "无正预测（TP+FP=0），精确率按 0 处理"
    else:
        res["precision"] = tp / (tp + fp)
        res["precision_reason"] = None
    if tp + fn == 0:
        res["recall"] = 0.0
        res["recall_reason"] = "实际无正例（TP+FN=0），召回率按 0 处理"
    else:
        res["recall"] = tp / (tp + fn)
        res["recall_reason"] = None
    if res["precision"] + res["recall"] == 0:
        res["f1"] = 0.0
        res["f1_reason"] = "精确率与召回率均为 0，F1 按 0 处理"
    else:
        res["f1"] = 2 * res["precision"] * res["recall"] / (res["precision"] + res["recall"])
        res["f1_reason"] = None
    return res


def _fmt_metric(value, reason=None, pct=False):
    if value is None:
        return "未定义"
    if pct:
        return f"{value:.2%}"
    return f"{value:.4f}"


ALARM_THRESHOLD_STRATEGIES = [
    ("min_cost", "最小总成本 (k·FN + FP)"),
    ("max_f1", "最大 F1"),
    ("recall_floor", "召回率下限约束 (recall ≥ 0.80 时精确率最高)"),
    ("manual", "手动指定"),
]


def scan_thresholds(proba, y, k_cost):
    """在验证集上扫描阈值 0.01~0.99，返回 (ths, rows)，每行含 threshold/precision/recall/f1/cost/fp/fn。"""
    ths = np.arange(0.01, 1.0, 0.01)
    rows = []
    for t in ths:
        mm = compute_metrics(proba, y, float(t))
        rows.append({
            "threshold": float(t),
            "precision": mm["precision"],
            "recall": mm["recall"],
            "f1": mm["f1"],
            "cost": k_cost * mm["false_negative"] + 1.0 * mm["false_positive"],
            "fp": mm["false_positive"],
            "fn": mm["false_negative"],
        })
    return ths, rows


def select_alarm_threshold(rows, strategy, k_cost, manual_value=None, recall_floor=0.80):
    """按策略选定报警阈值，返回 (threshold, reason_text, detail_dict)。"""
    costs = [r["cost"] for r in rows]
    f1s = [r["f1"] for r in rows]
    recalls = [r["recall"] for r in rows]

    if strategy == "min_cost":
        idx = int(np.argmin(costs))
        r = rows[idx]
        reason = (
            f"在验证集上扫描阈值 0.01~0.99，按总成本 = {k_cost:g}·FN + 1·FP 计算，"
            f"阈值 {r['threshold']:.2f} 时总成本最小 = {r['cost']:.0f}（FP={r['fp']}, FN={r['fn']}）。"
            f"该阈值在当前 k={k_cost:g} 下最兼顾误判与漏判的业务代价：k 越大越倾向降低阈值以提高召回。"
        )
        return r["threshold"], reason, r

    if strategy == "max_f1":
        idx = int(np.argmax(f1s))
        r = rows[idx]
        reason = (
            f"在验证集上扫描阈值，取 F1 = 2·P·R/(P+R) 最大处。"
            f"阈值 {r['threshold']:.2f} 时 F1 最大 = {r['f1']:.4f}"
            f"（精确率 {r['precision']:.4f}，召回率 {r['recall']:.4f}）。"
            f"F1 对误判与漏判同等加权，适用于两者业务代价相近的场景。"
        )
        return r["threshold"], reason, r

    if strategy == "recall_floor":
        candidates = [r for r in rows if r["recall"] >= recall_floor]
        if not candidates:
            idx = int(np.argmax(recalls))
            r = rows[idx]
            reason = (
                f"验证集上无阈值满足召回率 ≥ {recall_floor:.2f}，退化为取最大召回率阈值 "
                f"{r['threshold']:.2f}（召回率 {r['recall']:.4f}）。建议降低召回率下限或改用其他策略。"
            )
            return r["threshold"], reason, r
        r = max(candidates, key=lambda x: x["precision"])
        reason = (
            f"在召回率 ≥ {recall_floor:.2f} 的约束下，取精确率最高的阈值。"
            f"阈值 {r['threshold']:.2f} 时召回率 {r['recall']:.4f}、精确率 {r['precision']:.4f}"
            f"（FP={r['fp']}, FN={r['fn']}）。"
            f"适用于\"宁可误报不可漏报\"的高合规场景：先保证高风险客户被召回，再尽量减少误报。"
        )
        return r["threshold"], reason, r

    mv = float(manual_value) if manual_value is not None else 0.5
    closest = min(rows, key=lambda x: abs(x["threshold"] - mv))
    reason = (
        f"手动指定报警阈值 {mv:.2f}，对应验证集精确率 {closest['precision']:.4f}、"
        f"召回率 {closest['recall']:.4f}、F1 {closest['f1']:.4f}（FP={closest['fp']}, FN={closest['fn']}）。"
    )
    return mv, reason, closest


def compute_test_proba(model_key, bundle, data):
    """计算某模型在测试集上的违约概率向量。返回 ndarray 或 None（未知模型）。"""
    Xte = bundle["X_ready"]["test"]
    models = bundle["models"]
    if model_key in ("random_forest", "random_forest_calibrated"):
        return models[model_key].predict_proba(Xte)[:, 1]
    if model_key == "logistic_90dim":
        lg2 = models[model_key]
        return l2mod.predict_proba(Xte, lg2["w"], lg2["b"])
    if model_key == "logistic":
        df_test = data[data["split"] == "test"]
        return np.asarray(lcv2_new.predict_proba_apriori(df_test, models[model_key]))
    if model_key == "baseline":
        return np.zeros(len(bundle["y_test"]))
    return None


def evaluate_test_set(model_key, threshold, bundle, data):
    """对锁定的模型和阈值在测试集上计算一次指标。
    指标不能计算时填 reason（不填造数值）。
    返回 dict: model_key/threshold/n/tp/fp/fn/tn/precision/precision_reason/
              recall/recall_reason/auc/auc_reason/eval_error。
    """
    y_test = np.asarray(bundle["y_test"])
    result = {"model_key": model_key, "threshold": float(threshold), "n": int(len(y_test))}
    try:
        proba = compute_test_proba(model_key, bundle, data)
    except Exception as e:
        result["eval_error"] = f"测试集概率计算失败: {type(e).__name__}: {e}"
        return result
    if proba is None:
        result["eval_error"] = f"未知模型 {model_key}，无法计算测试集概率"
        return result

    m = compute_metrics(proba, y_test, float(threshold))
    result["tp"] = m["tp"]
    result["fp"] = m["false_positive"]
    result["fn"] = m["false_negative"]
    result["tn"] = m["tn"]
    result["precision"] = m["precision"]
    result["precision_reason"] = m["precision_reason"]
    result["recall"] = m["recall"]
    result["recall_reason"] = m["recall_reason"]

    try:
        result["auc"] = float(roc_auc_score(y_test, proba))
        result["auc_reason"] = None
    except ValueError as e:
        result["auc"] = None
        result["auc_reason"] = f"ROC-AUC 无法计算: {e}"
    return result


def render_test_eval(test_eval):
    """渲染单模型测试集评估结果：指标卡片 + 混淆矩阵，不能计算处显示原因。"""
    if "eval_error" in test_eval:
        st.error(test_eval["eval_error"])
        return

    def fmt_metric(value, reason):
        return f"{value:.4f}" if value is not None and reason is None else f"未定义：{reason}"

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("ROC-AUC", fmt_metric(test_eval.get("auc"), test_eval.get("auc_reason")))
    e2.metric("违约类精确率", fmt_metric(test_eval.get("precision"), test_eval.get("precision_reason")))
    e3.metric("召回率", fmt_metric(test_eval.get("recall"), test_eval.get("recall_reason")))
    e4.metric("误判数 (FP)", f"{test_eval['fp']:,}")
    e5.metric("漏判数 (FN)", f"{test_eval['fn']:,}")

    reasons = []
    if test_eval.get("auc_reason") is not None:
        reasons.append(f"ROC-AUC：{test_eval['auc_reason']}")
    if test_eval.get("precision_reason") is not None:
        reasons.append(f"精确率：{test_eval['precision_reason']}")
    if test_eval.get("recall_reason") is not None:
        reasons.append(f"召回率：{test_eval['recall_reason']}")
    if reasons:
        st.info("以下指标无法计算，已显示原因而非填造数值：\n- " + "\n- ".join(reasons))

    st.markdown("**混淆矩阵（行=真实，列=预测）**")
    cm_df = pd.DataFrame(
        {
            "预测非违约": [test_eval["tn"], test_eval["fn"]],
            "预测违约": [test_eval["fp"], test_eval["tp"]],
        },
        index=["实际非违约", "实际违约"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        f"测试集样本 {test_eval['n']:,} 条　|　"
        f"TP={test_eval['tp']}，FP={test_eval['fp']}（误判），"
        f"FN={test_eval['fn']}（漏判），TN={test_eval['tn']}"
    )


try:
    with st.spinner("正在加载数据..."):
        df = load_data(compute_data_sig())
except FileNotFoundError:
    st.error("数据文件未找到，请确认 01_credit_default.csv 在项目文件夹")
    st.stop()


load_locked_state_to_session()

st.title("客户风险分析系统")
st.markdown(
    '<p style="margin-top:-12px;font-size:1.05rem;color:#64748b;">'
    "基于信贷违约数据的风险建模与可视化分析平台</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 功能导航")
    st.markdown(
        '<hr style="margin:6px 0 14px 0;border:none;height:2px;'
        'background:linear-gradient(90deg,#4C78A8,transparent);">',
        unsafe_allow_html=True,
    )
    page = st.radio("页面选择", ["数据概览", "模型比较", "客户风险演示", "业务规则说明", "模型报警监控"])


def page_overview(data):
    st.header("数据概览")

    total = len(data)
    dup_record = int(data["record_id"].duplicated().sum())
    overall_default_rate = data["default_next_month"].mean()

    split_order = ["train", "valid", "test"]
    split_counts = data["split"].value_counts().reindex(split_order).fillna(0).astype(int)
    default_rates = data.groupby("split")["default_next_month"].mean().reindex(split_order).fillna(0)
    missing_by_split = (
        data.groupby("split")
        .apply(lambda g: int(g.isnull().sum().sum()))
        .reindex(split_order)
        .fillna(0)
        .astype(int)
    )

    st.subheader("一、统计卡片")
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    r1_c1.metric("总样本量", f"{total:,}")
    r1_c2.metric("record_id 重复数", f"{dup_record}")
    r1_c3.metric("总体违约率", f"{overall_default_rate:.2%}")

    st.markdown("**三组样本量**")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Train 样本量", f"{split_counts['train']:,}")
    sc2.metric("Valid 样本量", f"{split_counts['valid']:,}")
    sc3.metric("Test 样本量", f"{split_counts['test']:,}")

    st.markdown("**三组违约率**")
    dr1, dr2, dr3 = st.columns(3)
    dr1.metric("Train 违约率", f"{default_rates['train']:.2%}")
    dr2.metric("Valid 违约率", f"{default_rates['valid']:.2%}")
    dr3.metric("Test 违约率", f"{default_rates['test']:.2%}")

    st.markdown("**三组缺失值数**")
    ms1, ms2, ms3 = st.columns(3)
    ms1.metric("Train 缺失值", f"{missing_by_split['train']:,}")
    ms2.metric("Valid 缺失值", f"{missing_by_split['valid']:,}")
    ms3.metric("Test 缺失值", f"{missing_by_split['test']:,}")

    st.subheader("二、特征分布图")

    st.markdown("**1. 授信额度分布**")
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.hist(data["credit_limit_ntd"], bins=40, color="#38bdf8", edgecolor="#0a0f14", label="授信额度")
    ax1.set_xlabel("授信额度 (NTD)")
    ax1.set_ylabel("样本量")
    ax1.set_title("授信额度分布")
    ax1.legend(loc="upper right")
    st.pyplot(fig1)

    st.markdown("**2. 年龄分布**")
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.hist(data["age_years"], bins=range(int(data["age_years"].min()), int(data["age_years"].max()) + 2, 1), color="#fbbf24", edgecolor="#0a0f14", label="年龄")
    ax2.set_xlabel("年龄 (岁)")
    ax2.set_ylabel("样本量")
    ax2.set_title("客户年龄分布")
    ax2.legend(loc="upper right")
    st.pyplot(fig2)

    st.markdown("**3. 还款状态分布（按月分组）**")
    all_statuses = sorted(pd.unique(data[PAY_COLS].values.ravel()))
    x = np.arange(len(all_statuses))
    width = 0.13
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bar_palette = ["#5eead4", "#38bdf8", "#a78bfa", "#fbbf24", "#f472b6", "#34d399"]
    for i, col in enumerate(PAY_COLS):
        counts = data[col].value_counts().reindex(all_statuses, fill_value=0)
        ax3.bar(x + i * width, counts.values, width, color=bar_palette[i], edgecolor="#0a0f14", label=col.replace("pay_status_", ""))
    ax3.set_xlabel("还款状态编码")
    ax3.set_ylabel("样本量")
    ax3.set_title("各月还款状态分布")
    ax3.set_xticks(x + width * 2.5)
    ax3.set_xticklabels(all_statuses)
    ax3.legend(title="月份 (2005年)", loc="upper right")
    st.pyplot(fig3)

    st.subheader("三、还款状态编码说明")
    st.markdown(
        """
还款状态列 `pay_status_*` 的取值含义如下，**负值编码**表示未发生逾期：

| 编码 | 含义 |
|:---:|:---|
| **-2** | 无消费记录（账户激活但本月无账单） |
| **-1** | 按时全额还款 |
| **0** | 当期已还最低还款额 / 使用循环信用 |
| **1 ~ 9** | 逾期 1 ~ 9 个月 |

即：**-2、-1、0 均属于"未逾期"**，正值数字越大代表逾期越严重。
"""
    )

    st.subheader("四、敏感属性使用声明")
    st.markdown(
        """
数据集中 `sex_code`（性别）、`education_code`（教育）、`marriage_code`（婚姻）属于**敏感属性**。

> **相关性 ≠ 因果关系**：即便统计上某敏感属性与违约率存在相关，也不能据此推断该属性"导致"违约。
> 直接将敏感属性纳入风险模型可能引发**公平性与合规风险**（如歧视特定群体）。
> 本项目在建模与演示中仅作探索性分析，**不将敏感属性作为风险评分的因果依据**，必要时予以剔除或做公平性约束。
"""
    )


def page_demo(data):
    st.header("客户风险演示")

    locked = st.session_state.get("locked", False)
    demo_model_options = ["random_forest", "random_forest_calibrated", "logistic", "logistic_90dim"]
    locked_model = st.selectbox(
        "选择演示模型",
        demo_model_options,
        index=0,
        format_func=lambda k: MODEL_LABELS.get(k, k),
    )
    locked_threshold = get_locked_threshold(locked_model)
    model_locked = is_model_locked(locked_model)

    st.markdown(
        f"**当前使用模型**:{MODEL_LABELS.get(locked_model, locked_model)}　|　**当前阈值**:{locked_threshold:.2f}"
    )
    if model_locked:
        st.success(f"{MODEL_LABELS.get(locked_model, locked_model)} 已锁定阈值 {locked_threshold:.2f}（持久化于 {LOCKED_STATE_FILE}）。")
    else:
        st.info(f"{MODEL_LABELS.get(locked_model, locked_model)} 未显式锁定，使用推荐阈值 {locked_threshold:.2f}。可在【模型比较】页锁定后持久化。")

    with st.spinner("正在训练模型（首次加载较慢）"):
        bundle = train_models(compute_model_sig())
        models = bundle["models"]
        feature_cols = bundle["feature_cols"]
        X_ready = bundle["X_ready"]

    st.subheader("一、选择数据子集与记录")
    subset = st.selectbox("选择数据子集", ["train", "valid", "test"])
    sub_df = data[data["split"] == subset]
    id_min = int(sub_df["record_id"].min())
    id_max = int(sub_df["record_id"].max())
    valid_ids = set(sub_df["record_id"].tolist())
    max_id = int(data["record_id"].max())

    st.caption(f"{subset} 子集 record_id 范围：{id_min} ~ {id_max}（共 {len(sub_df):,} 条）；总体编号范围 0 ~ {max_id}")
    input_id = st.number_input(
        "输入或选择记录编号",
        value=id_min,
        step=1,
    )

    if int(input_id) < 0 or int(input_id) > max_id:
        st.error(f"编号超出范围，请输入 0 到 {max_id} 之间。")
        st.info("可从下方下拉框快速选择一条该子集中的记录。")
        quick_id = st.selectbox("或从子集中快速选择一条记录", options=list(sub_df["record_id"]))
        input_id = int(quick_id)
        st.success(f"已自动选用有效编号 {input_id}。")
    elif int(input_id) not in valid_ids:
        st.error(f"编号 {int(input_id)} 不在 {subset} 子集中（编号不连续），请输入该子集中存在的 record_id。")
        st.info("提示：可先用下方下拉框从该子集中快速选择一条记录。")
        quick_id = st.selectbox("或从子集中快速选择一条记录", options=list(sub_df["record_id"]))
        input_id = int(quick_id)
        st.success(f"已自动选用有效编号 {input_id}。")

    row = sub_df[sub_df["record_id"] == int(input_id)].iloc[0]

    st.subheader("二、该记录主要输入信息")
    info = {
        "record_id": int(row["record_id"]),
        "授信额度(NTD)": int(row["credit_limit_ntd"]),
        "年龄": int(row["age_years"]),
        "性别编码": int(row["sex_code"]),
        "教育编码": int(row["education_code"]),
        "婚姻编码": int(row["marriage_code"]),
    }
    for col in PAY_COLS:
        info[col.replace("pay_status_", "还款状态_")] = int(row[col])
    st.dataframe(pd.DataFrame([info]).T, use_container_width=True)

    st.subheader("三、违约概率预测")
    if locked_model == "baseline":
        prob = 0.0
        st.info("全判不违约基线：对所有客户输出违约概率 = 0。")
    elif locked_model in ("random_forest", "random_forest_calibrated"):
        row_idx = int(np.where((sub_df["record_id"].values == int(input_id)))[0][0])
        sample = X_ready[subset][row_idx]
        model = models[locked_model]
        prob = float(model.predict_proba(sample.reshape(1, -1))[0, 1])
        st.caption(f"{MODEL_LABELS[locked_model]} 加载自 joblib，使用 prepared_data.joblib 中该记录对应的 90 维特征工程数据预测。")
    elif locked_model == "logistic":
        model_new = models[locked_model]
        if model_new is None:
            st.error("逻辑回归(新版)未训练成功，无法预测。")
            return
        row_df = sub_df[sub_df["record_id"] == int(input_id)]
        try:
            prob = float(lcv2_new.predict_one_apriori(row_df, model_new))
        except (KeyError, ValueError) as e:
            st.error(f"该记录无法离散化，预测失败：{e}")
            return
        st.caption("逻辑回归 使用关联规则前件特征，经梯度下降 L2 逻辑回归预测。")
    elif locked_model == "logistic_90dim":
        lg2 = models[locked_model]
        row_idx = int(np.where((sub_df["record_id"].values == int(input_id)))[0][0])
        sample = X_ready[subset][row_idx]
        prob = float(l2mod.predict_proba(sample.reshape(1, -1), lg2["w"], lg2["b"])[0])
        st.caption("逻辑回归2 使用 prepared_data.joblib 中该记录对应的 90 维特征工程数据，经手写梯度下降 L2 逻辑回归预测。")
    else:
        lg2 = models[locked_model]
        row_idx = int(np.where((sub_df["record_id"].values == int(input_id)))[0][0])
        sample = X_ready[subset][row_idx]
        prob = float(l2mod.predict_proba(sample.reshape(1, -1), lg2["w"], lg2["b"])[0])

    rules_file = get_rules_file_for_model(locked_model) or BUSINESS_RULES_FILE
    rules, rules_status = load_business_rules(get_file_signature(rules_file), file_path=rules_file)
    final_decision = None
    if rules_status == "ok":
        st.subheader("四、分层决策（业务规则）")
        st.caption(f"规则文件：{rules_file}　|　模型：{MODEL_LABELS.get(locked_model, locked_model)}")
        feature_values = {}
        overdue_flag = 0
        feat_rules = rules.get("特征规则", {})
        if "逾期标志" in feat_rules:
            overdue_cols = feat_rules["逾期标志"]["特征列"]
            if locked_model in ("random_forest", "random_forest_calibrated", "logistic_90dim"):
                row_idx = int(np.where((sub_df["record_id"].values == int(input_id)))[0][0])
                sample = X_ready[subset][row_idx]
                for col in overdue_cols:
                    idx = int(col[1:])
                    if idx < len(sample):
                        feature_values[col] = float(sample[idx])
            overdue_flag, missing_flags = compute_overdue_flag(feature_values, overdue_cols)
            if missing_flags:
                st.warning(f"部分逾期标志特征缺失，已按 0 补全：{', '.join(missing_flags)}")
        elif locked_model in ("random_forest", "random_forest_calibrated", "logistic_90dim"):
            row_idx = int(np.where((sub_df["record_id"].values == int(input_id)))[0][0])
            sample = X_ready[subset][row_idx]
            directions = rules.get("核心特征方向", {})
            for feat in directions.keys():
                idx = int(feat[1:])
                if idx < len(sample):
                    feature_values[feat] = float(sample[idx])
        steps, final_decision = execute_layered_decision(prob, overdue_flag, rules, locked_threshold=locked_threshold)
        render_decision_steps(steps)
        render_final_decision_card(final_decision)

        st.subheader("五、概率规则与特征规则")
        render_probability_rules(rules, prob)
        if "逾期标志" in feat_rules:
            render_feature_rules(rules, overdue_flag)
        if "Top关联规则" in feat_rules:
            st.markdown("**Top 关联规则**")
            render_top_association_rules(rules)

        st.subheader("六、核心特征方向")
        render_core_feature_directions(rules, feature_values)
    else:
        st.warning(f"业务规则文件 {rules_file} 加载失败，已降级为简单阈值判断")
        is_high = prob >= locked_threshold
        p1, p2, p3 = st.columns(3)
        p1.metric("风险概率", f"{prob:.4f}")
        p2.metric("所选阈值", f"{locked_threshold:.2f}")
        if is_high:
            p3.metric("风险提示", "🔴 高风险")
        else:
            p3.metric("风险提示", "🟢 低风险")
        if is_high:
            st.error(f"该记录违约概率 {prob:.4f} ≥ 阈值 {locked_threshold:.2f}，判定为**高风险**。")
        else:
            st.success(f"该记录违约概率 {prob:.4f} < 阈值 {locked_threshold:.2f}，判定为**低风险**。")

    if subset == "test":
        st.subheader("七、事后真实标签（仅 test 子集）")
        true_label = int(row["default_next_month"])
        if final_decision is not None and final_decision["结论档位"] is not None:
            pred_label = 1 if final_decision["结论档位"] == "拒绝" else (0 if final_decision["结论档位"] == "通过" else None)
        else:
            pred_label = 1 if (prob >= locked_threshold) else 0
        t1, t2 = st.columns(2)
        if true_label == 1:
            t1.metric("真实标签", "1(违约)")
        else:
            t1.metric("真实标签", "0(未违约)")
        if pred_label is None:
            t2.metric("预测结果", "人工审核中")
            st.info("灰名单场景，暂不对比真实标签。")
        else:
            if pred_label == 1:
                t2.metric("预测结果", "1(高风险)")
            else:
                t2.metric("预测结果", "0(低风险)")
            if pred_label == true_label:
                st.success("预测与真实标签一致。")
            elif pred_label == 1 and true_label == 0:
                st.warning("误判(FP):将未违约客户判为高风险。")
            else:
                st.warning("漏判(FN)将违约客户判为低风险。")

    st.subheader("八、声明")
    st.markdown(
        "> ⚠️ 本页结果仅作**教学示例**使用，模型与阈值仅用于演示流程，**不得作为实际授信决定依据**。"
        "真实授信需结合更完整的特征、合规审查与人工复核。"
    )


def page_rules():
    st.header("业务规则说明")

    selected_model = st.selectbox(
        "选择模型查看对应业务规则",
        options=[m[0] for m in RULES_PAGE_MODEL_OPTIONS],
        format_func=lambda k: dict(RULES_PAGE_MODEL_OPTIONS).get(k, k),
    )
    rules_file = get_rules_file_for_model(selected_model) or BUSINESS_RULES_FILE
    rules, status = load_business_rules(get_file_signature(rules_file), file_path=rules_file)
    if status != "ok":
        st.error(f"业务规则文件 {rules_file} 加载失败，请检查文件是否存在且格式正确")
        return

    st.caption(f"当前规则文件：{rules_file}")

    st.subheader("一、模型信息")
    st.markdown(f"**模型名称**：{rules['模型名称']}")
    st.markdown(f"**模型文件**：{rules['模型文件']}")
    st.markdown(f"**阈值**：{rules['阈值']}")
    st.markdown(f"**说明**：{rules['说明']}")

    st.subheader("二、概率规则三档")
    render_probability_rules(rules, None)

    feat_rules = rules.get("特征规则", {})
    if "逾期标志" in feat_rules:
        st.subheader("三、逾期标志特征规则")
        render_feature_rules(rules, None)
    if "Top关联规则" in feat_rules:
        st.subheader("三、Top 关联规则")
        render_top_association_rules(rules)

    if rules.get("核心特征方向"):
        st.subheader("四、核心特征方向")
        render_core_feature_directions(rules, {})

    st.subheader("五、分层决策逻辑")
    for s in rules["分层决策逻辑"]:
        st.markdown(f"**步骤 {s['步骤']}**：{s['说明']}")

    st.subheader("六、补充信息")
    render_optional_fields(rules)


ALARM_CONFIG_FILE = "alarm_config.json"

ALARM_MODEL_NAME_MAP = {
    "随机森林(原始)": "随机森林",
    "随机森林(校准)": "随机森林(校准)",
    "关联规则+逻辑回归": "逻辑回归",
    "手写逻辑回归": "逻辑回归2",
}


def load_alarm_config():
    if not os.path.exists(ALARM_CONFIG_FILE):
        return None, "文件不存在"
    try:
        with open(ALARM_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f), "ok"
    except Exception as e:
        return None, str(e)


def _alarm_band_color(label: str) -> str:
    if "黄灯" in label:
        return "#fbbf24"
    if "红灯" in label:
        return "#f472b6"
    return "#5eead4"


def _render_alarm_band(title: str, band: dict):
    st.markdown(f"**{title}**")
    base = band["基线值"]
    y_lo, y_hi = band["黄灯下限"], band["黄灯上限"]
    r_lo, r_hi = band["红灯下限"], band["红灯上限"]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.barh([0], [r_hi - r_lo], left=[r_lo], color="#f472b6", alpha=0.35, label="红灯区间")
    ax.barh([0], [y_hi - y_lo], left=[y_lo], color="#fbbf24", alpha=0.35, label="黄灯区间")
    ax.axvline(base, color="#5eead4", linestyle="--", linewidth=2, label=f"基线={base:.4f}")
    ax.set_yticks([])
    ax.set_xlabel("占比 / 违约率")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("基线值", f"{base:.4f}")
    b2.metric("黄灯下限", f"{y_lo:.4f}")
    b3.metric("黄灯上限", f"{y_hi:.4f}")
    b4.metric("红灯下限", f"{r_lo:.4f}")
    b5.metric("红灯上限", f"{r_hi:.4f}")


def page_alarm_monitor():
    st.header("模型报警监控")

    cfg, status = load_alarm_config()
    if cfg is None:
        st.error(f"报警配置文件 {ALARM_CONFIG_FILE} 加载失败：{status}")
        return

    st.caption(f"配置文件：{ALARM_CONFIG_FILE}")

    st.subheader("一、配置概览")
    o1, o2, o3 = st.columns(3)
    o1.metric("版本号", cfg.get("版本号", "—"))
    o2.metric("锁定状态", "已锁定" if cfg.get("锁定状态") else "未锁定")
    o3.metric("模型数量", len(cfg.get("模型报警配置", {})))
    if cfg.get("说明"):
        st.info(cfg["说明"])

    st.subheader("二、容忍比例预设")
    preset = cfg.get("容忍比例预设", {})
    yellow_opts = preset.get("黄灯可选", [])
    red_opts = preset.get("红灯可选", [])
    cur_yellow = preset.get("当前黄灯", "—")
    cur_red = preset.get("当前红灯", "—")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**当前黄灯容忍比例**：{cur_yellow}")
        if yellow_opts:
            sel_y = st.selectbox("调整黄灯档位（仅提示，不自动写入配置）", yellow_opts, index=yellow_opts.index(cur_yellow) if cur_yellow in yellow_opts else 0)
            if sel_y != cur_yellow:
                st.caption(f"⚠ 若采用 {sel_y}，需重算黄灯/红灯区间并生成新版本配置文件。")
    with pc2:
        st.markdown(f"**当前红灯容忍比例**：{cur_red}")
        if red_opts:
            sel_r = st.selectbox("调整红灯档位（仅提示，不自动写入配置）", red_opts, index=red_opts.index(cur_red) if cur_red in red_opts else 0)
            if sel_r != cur_red:
                st.caption(f"⚠ 若采用 {sel_r}，需重算红灯区间并生成新版本配置文件。")

    st.subheader("三、触发规则")
    trigger = cfg.get("触发规则", {})
    for kind, rules in trigger.items():
        st.markdown(f"**{kind}**")
        for level, desc in rules.items():
            color = _alarm_band_color(level)
            st.markdown(
                f'<p style="border-left:4px solid {color};padding:6px 10px;'
                f'background:rgba(255,255,255,0.04);margin:4px 0;">'
                f'<b style="color:{color}">{level}</b>：{desc}</p>',
                unsafe_allow_html=True,
            )

    st.subheader("四、各模型报警配置")
    model_cfg = cfg.get("模型报警配置", {})
    if not model_cfg:
        st.warning("配置中未找到模型报警配置。")
    else:
        model_items = list(model_cfg.items())
        default_idx = 0
        sel_name = st.selectbox(
            "选择模型查看报警基线与区间",
            options=[m[0] for m in model_items],
            format_func=lambda k: ALARM_MODEL_NAME_MAP.get(k, k),
            index=default_idx,
        )
        m_cfg = model_cfg[sel_name]
        display_name = ALARM_MODEL_NAME_MAP.get(sel_name, sel_name)

        st.markdown(f"**模型**：{display_name}　|　**总样本数**：{m_cfg.get('总样本数', '—')}")

        cm = m_cfg.get("混淆矩阵", {})
        if cm:
            st.markdown("**混淆矩阵**")
            cm_df = pd.DataFrame(
                {
                    "预测非违约": [cm.get("TN", 0), cm.get("FN", 0)],
                    "预测违约": [cm.get("FP", 0), cm.get("TP", 0)],
                },
                index=["实际非违约", "实际违约"],
            )
            st.dataframe(cm_df, use_container_width=True)

        main_band = m_cfg.get("主报警_高风险占比")
        if main_band:
            st.subheader("主报警：高风险占比")
            _render_alarm_band("主报警_高风险占比", main_band)
            st.caption("主报警基于本周/本期高风险客户占比，连续 2 周越黄灯界→黄灯观察；连续 3 周或单周越红灯界→红灯并人工排查。")

        aux_band = m_cfg.get("辅报警_高风险组违约率")
        if aux_band:
            st.subheader("辅报警：高风险组违约率")
            _render_alarm_band("辅报警_高风险组违约率", aux_band)
            note = m_cfg.get("辅报警标注", "")
            if note:
                st.caption(f"辅报警标注：{note}")

    st.subheader("五、锁定理由")
    lock_reasons = cfg.get("锁定理由", {})
    if not lock_reasons:
        st.caption("配置未提供锁定理由。")
    else:
        for key, text in lock_reasons.items():
            locked = key.endswith("_锁定") or key == "整体锁定原则"
            icon = "🔒" if locked else "🔓"
            badge = "锁定" if locked else "可调"
            color = "#f472b6" if locked else "#5eead4"
            st.markdown(
                f'<p style="border-left:4px solid {color};padding:6px 10px;'
                f'background:rgba(255,255,255,0.04);margin:6px 0;">'
                f'{icon} <b style="color:{color}">[{badge}] {key}</b><br>{text}</p>',
                unsafe_allow_html=True,
            )


def page_compare():
    st.header("模型比较")

    with st.spinner("正在训练模型（首次加载较慢）"):
        bundle = train_models(compute_model_sig())
    valid_proba = bundle["valid_proba"]
    valid_y = bundle["valid_y"]

    model_key = st.selectbox("选择模型", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    threshold = st.slider("阈值", 0.01, 0.99, 0.5, 0.01)
    k_cost = st.number_input("漏判成本倍数 k（总成本 = k·FN + 1·FP）", min_value=0.0, value=5.0, step=0.5)
    render_recommend_hint(k_cost, model_key)
    with st.expander("参数 k 的含义", expanded=False):
        st.markdown(
            """
            **业务成本函数**：`总成本 = k·FN + 1·FP`

            - **FN（假阴性 / 漏判）**：实际为高风险客户，却被模型判为低风险。后果是错失风险客户，可能造成坏账损失。
            - **FP（假阳性 / 误判）**：实际为低风险客户，却被模型判为高风险。后果是误拒正常客户，损失业务机会。
            - **k**：漏判一次高风险客户相对于误判一次低风险客户的代价倍数。

            **默认 k = 5 的含义**：在客户风险分析中，漏报一个真正的高风险客户（FN）造成的损失约为误报一个低风险客户（FP）的 **5 倍**，即"漏报"远比"误报"严重。

            - k 越大，越倾向于**宁可误判也要避免漏判**（提高召回率，降低阈值）；
            - k 越小（接近 0），越倾向于**宁可漏判也要避免误判**（提高精确率，提高阈值）；
            - k = 1 时，漏判与误判代价相同，等价于直接最小化错误总数 FN + FP。

            可在上方输入框调整 k，页面将据此重新计算总成本并标注最小成本阈值。
            """
        )

    st.subheader("一、验证集指标（当前模型 + 当前阈值）")
    if valid_proba[model_key] is None:
        st.warning(f"{MODEL_LABELS[model_key]} 加载自 joblib，需 90 维特征工程数据，仅支持单条演示（见【客户风险演示】页），无法批量评估验证集。")
    else:
        m = compute_metrics(valid_proba[model_key], valid_y, threshold)
        total_cost = k_cost * m["false_negative"] + 1.0 * m["false_positive"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("精确率", _fmt_metric(m["precision"]))
        c2.metric("召回率", _fmt_metric(m["recall"]))
        c3.metric("F1", _fmt_metric(m["f1"]))
        c4.metric("误判数 (FP)", f"{m['false_positive']:,}")
        c5.metric("漏判数 (FN)", f"{m['false_negative']:,}")
        c6.metric(f"总成本 (k={k_cost:g})", f"{total_cost:,.1f}")
        reasons = []
        if m["precision_reason"] is not None:
            reasons.append(f"精确率：{m['precision_reason']}")
        if m["recall_reason"] is not None:
            reasons.append(f"召回率：{m['recall_reason']}")
        if m["f1_reason"] is not None:
            reasons.append(f"F1：{m['f1_reason']}")
        if reasons:
            st.info("以下指标按 0 处理：\n- " + "\n- ".join(reasons))
        else:
            st.caption("所有指标均可计算。")

    st.subheader("二、阈值变化曲线（验证集）")
    scan_rows = None
    if valid_proba[model_key] is None:
        st.info(f"{MODEL_LABELS[model_key]} 无法批量评估验证集，阈值变化曲线不可用。")
    else:
        ths, scan_rows = scan_thresholds(valid_proba[model_key], valid_y, k_cost)
        prec = [r["precision"] for r in scan_rows]
        rec = [r["recall"] for r in scan_rows]
        f1 = [r["f1"] for r in scan_rows]
        cost = np.asarray([r["cost"] for r in scan_rows])
        best_idx = int(np.argmin(cost))
        best_th_cost = float(ths[best_idx])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ths, prec, label="精确率", color="#38bdf8")
        ax.plot(ths, rec, label="召回率", color="#fbbf24")
        ax.plot(ths, f1, label="F1", color="#5eead4")
        ax.axvline(threshold, color="#f472b6", linestyle="--", label=f"当前阈值={threshold:.2f}")
        ax.axvline(best_th_cost, color="#34d399", linestyle=":", label=f"最小成本阈值={best_th_cost:.2f}")
        ax.set_xlabel("阈值")
        ax.set_ylabel("指标值")
        ax.set_title(f"{MODEL_LABELS[model_key]} - 阈值变化曲线（验证集）")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(ths, cost, label=f"总成本 = {k_cost:g}·FN + 1·FP", color="#f472b6")
        ax2.axvline(threshold, color="#f472b6", linestyle="--", label=f"当前阈值={threshold:.2f}")
        ax2.axvline(best_th_cost, color="#34d399", linestyle=":", label=f"最小成本阈值={best_th_cost:.2f}")
        ax2.scatter([best_th_cost], [cost[best_idx]], color="#34d399", zorder=5)
        ax2.set_xlabel("阈值")
        ax2.set_ylabel("总成本")
        ax2.set_title(f"{MODEL_LABELS[model_key]} - 阈值-总成本曲线（验证集，k={k_cost:g}）")
        ax2.legend(loc="best")
        st.pyplot(fig2)
        st.caption(f"最小总成本 = {cost[best_idx]:,.1f}，对应阈值 = {best_th_cost:.2f}；当前阈值成本 = {k_cost * m['false_negative'] + 1.0 * m['false_positive']:,.1f}。")
        if model_key == "baseline":
            st.caption("基线模型输出恒为 0：无正预测，精确率与 F1 按 0 处理，召回率=0。")

    st.subheader("二·2、报警阈值选定")
    if scan_rows is None:
        st.info(f"{MODEL_LABELS[model_key]} 无法批量评估验证集，报警阈值选定不可用。")
    else:
        strategy = st.selectbox(
            "报警阈值选定策略",
            options=[s[0] for s in ALARM_THRESHOLD_STRATEGIES],
            format_func=lambda k: dict(ALARM_THRESHOLD_STRATEGIES)[k],
        )
        manual_value = None
        recall_floor = 0.80
        if strategy == "manual":
            manual_value = st.slider("手动指定报警阈值", 0.01, 0.99, float(threshold), 0.01)
        if strategy == "recall_floor":
            recall_floor = st.slider("召回率下限", 0.50, 0.99, 0.80, 0.01)

        alarm_th, alarm_reason, alarm_detail = select_alarm_threshold(
            scan_rows, strategy, k_cost, manual_value=manual_value, recall_floor=recall_floor
        )

        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        ac1.metric("选定报警阈值", f"{alarm_th:.2f}")
        ac2.metric("精确率", f"{alarm_detail['precision']:.4f}")
        ac3.metric("召回率", f"{alarm_detail['recall']:.4f}")
        ac4.metric("F1", f"{alarm_detail['f1']:.4f}")
        ac5.metric(f"总成本 (k={k_cost:g})", f"{alarm_detail['cost']:,.1f}")

        with st.expander("选定依据（为什么是这个阈值）", expanded=True):
            st.markdown(alarm_reason)
            st.markdown(
                f"对应验证集混淆：FP（误判）={alarm_detail['fp']}，"
                f"FN（漏判）={alarm_detail['fn']}。"
            )
            if strategy == "min_cost":
                st.markdown(
                    "**适用场景**：已明确漏判与误判的业务代价比例 k，追求总业务成本最低。"
                    "k 增大时最优阈值下移（更激进报警），k 减小时上移（更保守）。"
                )
            elif strategy == "max_f1":
                st.markdown(
                    "**适用场景**：漏判与误判代价相近（k≈1），追求精确率与召回率的调和均值最高。"
                )
            elif strategy == "recall_floor":
                st.markdown(
                    f"**适用场景**：合规要求高风险客户召回率不低于 {recall_floor:.2f}，"
                    "在该约束下尽量减少误报以降低审核负担。"
                )
            else:
                st.markdown(
                    "**适用场景**：业务专家凭经验直接指定阈值，需人工对齐外部政策或历史口径。"
                )

        col_apply, col_show = st.columns([1, 2])
        with col_apply:
            if st.button("应用为报警阈值（同步到客户风险演示页）"):
                mts = dict(st.session_state.get("model_thresholds", {}) or {})
                mts[model_key] = float(alarm_th)
                st.session_state.model_thresholds = mts
                st.session_state.locked = True
                save_locked_state()
                st.success(f"已锁定 {MODEL_LABELS.get(model_key, model_key)} 报警阈值 {alarm_th:.2f}，客户风险演示页该模型将使用此阈值。")
        with col_show:
            mts = st.session_state.get("model_thresholds", {}) or {}
            if mts:
                rows = [{"模型": MODEL_LABELS.get(k, k), "锁定阈值": f"{v:.2f}"} for k, v in mts.items()]
                st.caption("已锁定阈值（持久化）：")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("尚未锁定任何模型阈值。")

    if model_key == "baseline":
        st.subheader("二·5、基线完整评估（复用 logistic_credit_v2.baseline_all_no_default）")
        y_test = bundle["y_test"]
        bl = lcv2.baseline_all_no_default(valid_y, y_test, threshold)
        for name, ev_b, auc_b, y_true in [("valid", bl["eval_valid"], bl["auc_valid"], valid_y),
                                          ("test", bl["eval_test"], bl["auc_test"], y_test)]:
            non_default_rate = 1.0 - np.mean(y_true)
            st.markdown(f"**[{name}]** 样本数 {len(y_true)}　实际违约 {int(np.sum(y_true))} 人　非违约率 {non_default_rate:.4f}")
            cm_df = pd.DataFrame(
                {
                    "预测非违约": [ev_b["tn"], ev_b["fn"]],
                    "预测违约": [ev_b["fp"], ev_b["tp"]],
                },
                index=["实际非违约", "实际违约"],
            )
            st.dataframe(cm_df, use_container_width=True)
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("准确率", f"{ev_b['acc']:.4f}")
            mc2.metric("精确率", f"{ev_b['prec']:.4f}")
            mc3.metric("召回率", f"{ev_b['rec']:.4f}")
            mc4.metric("F1", f"{ev_b['f1']:.4f}")
            mc5.metric("AUC", f"{auc_b:.4f}")
            ok = abs(ev_b["acc"] - non_default_rate) < 1e-9
            st.caption(f"误判(非违约→违约) {ev_b['fp']} 人，漏判(违约→非违约) {ev_b['fn']} 人　|　校验: 准确率 == 非违约率 {'✓' if ok else '✗'}")

    st.subheader("三、模型对比（验证集 AUC）")
    rows = []
    for k in list(MODEL_LABELS.keys()):
        if valid_proba[k] is None:
            rows.append({"模型": MODEL_LABELS[k], "验证集 AUC": "无法批量计算", "说明": "加载自 joblib，需 90 维特征工程，仅支持单条演示"})
            continue
        try:
            auc = roc_auc_score(valid_y, valid_proba[k])
            auc_str = f"{auc:.4f}"
            note = "常数预测，AUC=0.5 无区分能力" if k == "baseline" else "—"
        except Exception as e:
            auc_str = "无法计算"
            note = str(e)
        rows.append({"模型": MODEL_LABELS[k], "验证集 AUC": auc_str, "说明": note})
    st.table(pd.DataFrame(rows))

    st.subheader("四、锁定并评价测试集")
    st.caption(f"锁定状态持久化于 `{LOCKED_STATE_FILE}`，换电脑/重启后仍保留。每模型独立锁定各自阈值。")
    mts = dict(st.session_state.get("model_thresholds", {}) or {})
    cur_model_locked = model_key in mts
    cur_model_th = mts.get(model_key, threshold)

    st.markdown(
        f"**当前模型**：{MODEL_LABELS.get(model_key, model_key)}　|　"
        f"**当前阈值**：{threshold:.2f}　|　"
        f"**该模型已锁定阈值**：{cur_model_th:.2f}{'（已锁定）' if cur_model_locked else '（未锁定，=当前阈值）'}"
    )

    if not cur_model_locked:
        if st.button(f"锁定 {MODEL_LABELS.get(model_key, model_key)} 当前阈值并评价测试集"):
            mts[model_key] = float(threshold)
            st.session_state.model_thresholds = mts
            st.session_state.locked = True
            st.session_state.pop("test_eval_cache", None)
            save_locked_state()
            st.rerun()
    else:
        lm = model_key
        lt = cur_model_th
        st.success(f"已锁定 {MODEL_LABELS.get(lm, lm)} 阈值={lt:.2f}。测试集评价如下（只计算一次，结果缓存于会话）。")

        cache_key = (lm, round(float(lt), 4), compute_model_sig())
        cached = st.session_state.get("test_eval_cache")
        if cached is None or cached.get("key") != cache_key:
            with st.spinner("正在测试集上评价锁定模型…"):
                test_eval = evaluate_test_set(lm, lt, bundle, df)
            test_eval["key"] = cache_key
            st.session_state.test_eval_cache = test_eval
        else:
            test_eval = cached
            st.caption("（使用会话内缓存结果，未重复计算）")

        render_test_eval(test_eval)

        if st.button(f"解除 {MODEL_LABELS.get(model_key, model_key)} 锁定"):
            mts.pop(model_key, None)
            st.session_state.model_thresholds = mts
            st.session_state.locked = len(mts) > 0
            st.session_state.pop("test_eval_cache", None)
            save_locked_state()
            st.rerun()

    st.markdown("**已锁定阈值汇总**")
    if mts:
        rows = [{"模型": MODEL_LABELS.get(k, k), "锁定阈值": f"{v:.2f}"} for k, v in mts.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("尚无模型被锁定。")


if page == "数据概览":
    page_overview(df)
elif page == "模型比较":
    page_compare()
elif page == "客户风险演示":
    page_demo(df)
elif page == "业务规则说明":
    page_rules()
elif page == "模型报警监控":
    page_alarm_monitor()