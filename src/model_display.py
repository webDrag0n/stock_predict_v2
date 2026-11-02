"""集成子模型的用户可见显示名（说人话层）。

内部标识符（配置键 ensemble_models、parquet 列 pred_<model>、weights dict 键、
manifest、指纹计算）一律用原始模型键，本模块不做任何改动；这里只定义
**显示层**的键 → 中文名映射，供 pipeline 步骤摘要与 webapp 渲染共用。

放在 src/ 而非 webapp/ 的原因：pipeline/steps_*（不依赖 dash）也要把人话文案
写进落盘的 step report；本模块零三方依赖，双向 import 都不会成环。
"""

from __future__ import annotations

import re

# 模型键 → (显示名, 一句话解释)。未知键不在表内：model_label 原样返回键名。
MODEL_DISPLAY: dict[str, tuple[str, str]] = {
    "gbm_l2": (
        "AI 树模型",
        "LightGBM 梯度提升树：非线性 AI 模型，从 300+ 因子自动学习规律",
    ),
    "gbm_ret_weighted": (
        "AI 树模型（收益加权）",
        "LightGBM 梯度提升树，训练样本按当日前瞻收益排名加权（与回测口径一致）",
    ),
    "ic_linear": (
        "因子加权模型",
        "按各因子近期预测准确率（IC）加权的线性打分",
    ),
    "attn": (
        "注意力模型",
        "注意力机制神经网络：自动学习因子间的时序交互模式",
    ),
}

# 长键在前，避免短键抢先匹配（交替正则的防御性写法）
_KEYS = sorted(MODEL_DISPLAY, key=len, reverse=True)
_KEY_ALT = "|".join(re.escape(k) for k in _KEYS)
# 「gbm_l2=0.50」→「AI 树模型 50%」；只在等号右侧是数字时按权重百分比转换
_KEY_EQ_NUM_RE = re.compile(rf"(?<![\w])({_KEY_ALT})=([0-9]*\.?[0-9]+)")
# 裸键（整词）→ 显示名。\w 边界保证 pred_gbm_l2 这类列名不被误改
_KEY_WORD_RE = re.compile(rf"(?<![\w])({_KEY_ALT})(?![\w])")


def model_label(key) -> str:
    """模型键 → 用户可见显示名；未知键原样返回（不丢信息）。"""
    entry = MODEL_DISPLAY.get(str(key))
    return entry[0] if entry else str(key)


def model_blurb(key) -> str:
    """模型键 → 一句话解释；未知键返回空串。"""
    entry = MODEL_DISPLAY.get(str(key))
    return entry[1] if entry else ""


def humanize_model_text(text) -> str:
    """把一段用户可见文本里的模型键换成显示名。

    两类替换（按序）：
      1. ``gbm_l2=0.50`` 这类「键=小数」权重写法 → 「AI 树模型 50%」；
      2. 其余裸键（整词）→ 显示名，如「权重 gbm_l2,ic_linear」→「权重 AI 树模型,因子加权模型」。

    非字符串/空值原样返回；不含已知键的文本零开销透传。
    """
    if not isinstance(text, str) or not text:
        return text

    def _pct(m: re.Match) -> str:
        try:
            return f"{model_label(m.group(1))} {float(m.group(2)):.0%}"
        except ValueError:
            return model_label(m.group(1))

    out = _KEY_EQ_NUM_RE.sub(_pct, text)
    return _KEY_WORD_RE.sub(lambda m: model_label(m.group(1)), out)
