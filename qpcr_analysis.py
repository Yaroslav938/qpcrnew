"""
qpcr_analysis.py

Высокоуровневый анализ qPCR поверх qpcr_data и qpcr_models:

- batch_fit: пакетный фиттинг всех проб в QPCRDataset
             (L4, L5 или авто‑выбор по AICc, как mselect/getPar в qpcR).[file:1][web:55]

- calib_efficiency: оценка эффективности по калибровочному ряду
                    Ct vs log10(dilution), формула E = 10^(-1/slope)[web:63].

- relative_expression: простая относительная экспрессия по (E^ΔCt)
                       как базовый аналог ratiocalc/refmean.[file:1][web:64]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple, Optional

import numpy as np
import pandas as pd

from qpcr_data import QPCRDataset
from qpcr_models import (
    FitResult,
    fit_curve_l4,
    fit_curve_l5,
    fit_curve_auto,
)


ModelChoice = Literal["L4", "L5", "auto"]
Criterion = Literal["AICc", "AIC", "R2"]


# ==========================
# ПАКЕТНЫЙ ФИТТИНГ
# ==========================

@dataclass
class BatchFitResult:
    """
    Результаты пакетного анализа:

    table      : DataFrame, 1 строка = 1 проба (колонка из QPCRDataset)
                 колонки: model, Ct_cpD2, Efficiency, RSS, R2, AIC, AICc и параметры L4/L5.

    fits       : словарь {sample_name: FitResult} для дальнейших детальных разборов.
    """

    table: pd.DataFrame
    fits: Dict[str, FitResult]


def batch_fit(
    dataset: QPCRDataset,
    model: ModelChoice = "auto",
    criterion: Criterion = "AICc",
    min_points: int = 8,
) -> BatchFitResult:
    """
    Пакетный фиттинг всех проб из QPCRDataset.

    model:
      - "L4"  : всегда подгонять L4
      - "L5"  : всегда L5
      - "auto": L4 vs L5, выбрать по criterion (по умолчанию AICc минимальный),
                аналог qpcR::mselect(do.all=TRUE)[file:1].

    Возвращает BatchFitResult с таблицей и словарём FitResult по пробам.
    """
    x = dataset.df[dataset.cycle_col].values

    rows = []
    fit_map: Dict[str, FitResult] = {}

    for col in dataset.sample_cols:
        y = dataset.df[col].values

        # Выбор модели
        if model == "L4":
            fit_res = fit_curve_l4(x, y)
        elif model == "L5":
            fit_res = fit_curve_l5(x, y)
        else:  # "auto"
            fit_res = fit_curve_auto(x, y, criterion=criterion)

        fit_map[col] = fit_res

        # Заполняем строку таблицы
        row = {
            "sample": col,
            "fit_ok": fit_res.success,
            "message": fit_res.message,
            "model": fit_res.model,
            "Ct_cpD2": fit_res.cpD2,
            "Efficiency_cpD2": fit_res.efficiency,
            "RSS": fit_res.rss,
            "R2": fit_res.r2,
            "AIC": fit_res.aic,
            "AICc": fit_res.aicc,
        }

        # Параметры модели по колонкам (b, c, d, e, f)
        for p_name in ["b", "c", "d", "e", "f"]:
            row[p_name] = fit_res.params.get(p_name, np.nan)

        rows.append(row)

    table = pd.DataFrame(rows)
    return BatchFitResult(table=table, fits=fit_map)


# ==========================
# КАЛИБРОВКА ЭФФЕКТИВНОСТИ (calib)
# ==========================

@dataclass
class CalibrationResult:
    """
    Результаты калибровки (аналог qpcR::calib без бутстрапа)[web:63]:

    slope       : наклон регрессии Ct ~ log10(dilution)
    intercept   : свободный член
    efficiency  : E = 10^(-1/slope)
    r2          : коэффициент детерминации
    """
    slope: float
    intercept: float
    efficiency: float
    r2: float


def calib_efficiency(
    ct_values: np.ndarray,
    dilutions: np.ndarray,
    log_base: float = 10.0,
) -> CalibrationResult:
    """
    Расчёт эффективности по калибровочному ряду, как в документации qpcR::calib[web:63]:

    - строим линейную регрессию Ct = a + b * log10(dilution)
    - E = log_base^(-1 / b)

    Параметры:
      ct_values : массив Ct (одна кривая на каждую точку разведения)
      dilutions : массив концентраций или коэффициентов разведения ( > 0 )

    ВАЖНО: порядок элементов ct_values и dilutions должен совпадать.
    """
    ct = np.asarray(ct_values, dtype=float)
    dil = np.asarray(dilutions, dtype=float)

    mask = np.isfinite(ct) & np.isfinite(dil) & (dil > 0)
    ct, dil = ct[mask], dil[mask]

    if len(ct) < 2:
        raise ValueError("Для калибровки нужно минимум две точки разведения.")

    logd = np.log10(dil) if log_base == 10.0 else np.log(dil) / np.log(log_base)

    # простая линейная регрессия (МНК)
    X = np.vstack([np.ones_like(logd), logd]).T
    beta, *_ = np.linalg.lstsq(X, ct, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])

    # эффективность
    efficiency = float(log_base ** (-1.0 / slope))

    # R²
    ct_hat = intercept + slope * logd
    ss_res = float(np.sum((ct - ct_hat) ** 2))
    ss_tot = float(np.sum((ct - ct.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        efficiency=efficiency,
        r2=r2,
    )


# ==========================
# ОТНОСИТЕЛЬНАЯ ЭКСПРЕССИЯ (упрощённый ratiocalc)
# ==========================

@dataclass
class RatioResult:
    """
    Упрощённая относительная экспрессия (без error propagation):

    ratio         : относительная экспрессия (target vs reference)
    log2_ratio    : log2(ratio)
    """
    ratio: float
    log2_ratio: float


def relative_expression(
    ct_target: float,
    ct_ref: float,
    eff_target: float,
    eff_ref: float,
    mode: Literal["deltaCt", "ddCt"] = "deltaCt",
    ct_ref_control: Optional[float] = None,
    ct_target_control: Optional[float] = None,
) -> RatioResult:
    """
    Базовый аналог qpcR::ratiocalc/refmean для одной пары target/reference[web:64][file:1].

    Поддерживаются два популярных варианта:

    mode = "deltaCt":
        ratio = (eff_target ** (-ct_target)) / (eff_ref ** (-ct_ref))

    mode = "ddCt":
        требуется ct_ref_control и ct_target_control (контрольная группа):
        ΔCt(sample)  = ct_target - ct_ref
        ΔCt(control) = ct_target_control - ct_ref_control
        ΔΔCt         = ΔCt(sample) - ΔCt(control)
        ratio        = eff_target ** (-ΔΔCt)
        (предполагаем одинаковую эффективность для target/ref, как в 2^-ΔΔCt).

    Возвращает RatioResult с ratio и log2(ratio).
    """
    if mode == "deltaCt":
        # относительная экспрессия в одной группе
        num = eff_target ** (-ct_target)
        den = eff_ref ** (-ct_ref)
        ratio = float(num / den)
    elif mode == "ddCt":
        if ct_ref_control is None or ct_target_control is None:
            raise ValueError("Для режима 'ddCt' нужно указать ct_ref_control и ct_target_control.")
        dct_sample = ct_target - ct_ref
        dct_control = ct_target_control - ct_ref_control
        ddct = dct_sample - dct_control
        # обычно используют обычную эффективность 2.0; здесь используем eff_target
        ratio = float(eff_target ** (-ddct))
    else:
        raise ValueError(f"Неизвестный режим относительной экспрессии: {mode}")

    log2_ratio = float(np.log2(ratio)) if ratio > 0 else np.nan
    return RatioResult(ratio=ratio, log2_ratio=log2_ratio)
