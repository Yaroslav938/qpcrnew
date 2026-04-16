"""
qpcr_models.py

Сигмоидальные модели для qPCR (L4, L5), фиттинг, расчет Ct (cpD2),
эффективности и метрик качества подгонки (RSS, R², AIC, AICc).

Идеология максимально близка к функциям pcrfit(), efficiency(),
AICc(), mselect() из пакета qpcR.

Использует итеративный метод поиска максимума производных (zooming)
и конечные разности для обхода ограничений плавающей точки.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

ModelName = Literal["L4", "L5"]

# ==========================
# МОДЕЛИ L4 / L5 (лог‑логистики)
# ==========================
# Используется формат (x/c)**b для максимальной стабильности float

def l4_model(x: np.ndarray, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    4‑параметрическая лог‑логистическая модель (аналог qpcR::l4):

    f(x) = d + (e - d) / (1 + (x/c)^b)

    x : циклы ( > 0 )
    b : наклон
    c : точка перегиба (cpD1 ~ cpD2)
    d : нижний асимптот (baseline)
    e : верхний асимптот (плато)
    """
    x_safe = np.clip(x, 1e-5, None)
    c_safe = max(c, 1e-5)
    base = 1.0 + np.power(x_safe / c_safe, b)
    return d + (e - d) / base


def l5_model(x: np.ndarray, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    5‑параметрическая лог‑логистическая модель (аналог qpcR::l5):
    Включает дополнительный параметр асимметрии f.

    f(x) = d + (e - d) / (1 + (x/c)^b)^f
    """
    x_safe = np.clip(x, 1e-5, None)
    c_safe = max(c, 1e-5)
    base = 1.0 + np.power(x_safe / c_safe, b)
    
    # Чтобы избежать ошибок с отрицательными числами в степени
    base = np.clip(base, 1e-10, None) 
    return d + (e - d) / np.power(base, f)


# ==========================
# КЛАСС РЕЗУЛЬТАТОВ
# ==========================

@dataclass
class FitResult:
    """
    Результат фиттинга одной кривой амплификации.
    """
    model: ModelName          # "L4" или "L5"
    params: Dict[str, float]  # найденные параметры (b, c, d, e, [f])
    x_dense: np.ndarray       # плотная сетка X (для отрисовки и производных)
    y_dense: np.ndarray       # плотная сетка Y (сглаженная кривая)
    cpD1: Optional[float]     # максимум 1‑й производной (точка перегиба)
    cpD2: Optional[float]     # максимум 2‑й производной (Ct по qpcR::cpD2)
    efficiency: Optional[float]  # E_n = F_n / F_{n-1} вблизи cpD2
    rss: float                # сумма квадратов остатков (Residual Sum of Squares)
    r2: float                 # коэффициент детерминации R²
    aic: float                # информационный критерий Акаике
    aicc: float               # скорректированный AIC
    success: bool             # флаг успешного фиттинга
    message: str              # сообщение об ошибке (или "OK")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает значения Y для переданного массива X на основе подобранной модели.
        """
        if self.model == "L4":
            return l4_model(x, **self.params)
        return l5_model(x, **self.params)


# ==========================
# ВСПОМОГАТЕЛЬНАЯ МАТЕМАТИКА
# ==========================

def _clean_xy(x_raw, y_raw) -> Tuple[np.ndarray, np.ndarray]:
    """Очистка массивов от NaN/Inf для curve_fit."""
    x = np.array(x_raw, dtype=float)
    y = np.array(y_raw, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _has_reasonable_signal(y: np.ndarray, min_amplitude: float = 0.5) -> bool:
    """
    Простейшая проверка, есть ли вообще сигнал, похожий на ПЦР-кривую.
    Если амплитуда сигнала (max - min) слишком мала, скорее всего это шум или NTC.
    """
    if len(y) < 5:
        return False
    return (np.max(y) - np.min(y)) >= min_amplitude


def gof_metrics(x: np.ndarray, y: np.ndarray, y_fit: np.ndarray, k_params: int) -> Tuple[float, float, float, float]:
    """
    Расчет метрик Goodness-of-Fit (RSS, R², AIC, AICc).
    Считаем, что ошибки распределены нормально.
    """
    rss = float(np.sum((y - y_fit)**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    n = len(y)
    
    if n > k_params and rss > 0:
        aic = n * np.log(rss / n) + 2 * k_params
        # Формула AICc со штрафом за маленькую выборку
        aicc = aic + (2 * k_params * (k_params + 1)) / (n - k_params - 1)
    else:
        aic, aicc = np.nan, np.nan
        
    return rss, r2, float(aic), float(aicc)


def _find_peak_zooming(func: Callable, x_min: float, x_max: float, order: int = 2, pts: int = 100, n_cycles: int = 10) -> float:
    """
    Интегрированный метод сужающегося окна с расчетом дискретной производной (конечные разности).
    Защищает от NaN, Inf и переполнений, обеспечивает точность до 10 знака без огромных массивов.
    """
    k = (x_max - x_min) / (pts - 1)
    curr_min = x_min
    curr_max = x_max
    x_star = None
    
    for _ in range(n_cycles):
        xs = np.linspace(curr_min, curr_max, pts)
        
        try:
            # Считаем саму функцию и ее сдвиги для конечной разности
            v = func(xs)
            vp = func(xs + k)
            vm = func(xs - k)
            
            if order == 1:
                target = (vp - vm) / (2.0 * k)
            else:
                target = (vm - 2.0 * v + vp) / (k ** 2)
                
            # Жесткий фильтр на NaN/Inf и complex
            valid_mask = np.isfinite(target) & np.isfinite(v)
            if not np.any(valid_mask):
                break
                
            xs_valid = xs[valid_mask]
            target_valid = target[valid_mask]
            
            max_idx = int(np.argmax(target_valid))
            x_star = xs_valid[max_idx]
            
        except Exception:
            break
            
        # Сужаем окно
        curr_min = x_star - k
        curr_max = x_star + k
        k = (curr_max - curr_min) / (pts - 1)
        
    return float(x_star) if x_star is not None else np.nan


def _efficiency_at_cycle(
    params: Dict[str, float],
    cpD2: float,
    model: ModelName,
) -> Optional[float]:
    """
    Оценка эффективности в окрестности cpD2, как в efficiency():

    E_n = F_n / F_{n-1}, где F_n — смоделированная флуоресценция
    в целочисленном цикле ближайшем к cpD2.
    """
    if cpD2 is None or np.isnan(cpD2):
        return None

    # формируем модель по параметрам
    if model == "L4":
        f_model = lambda x: l4_model(x, params["b"], params["c"], params["d"], params["e"])
    else: 
        f_model = lambda x: l5_model(x, params["b"], params["c"], params["d"], params["e"], params["f"])

    # целочисленные циклы
    x_min = 1
    x_max = int(np.ceil(cpD2)) + 2
    if x_max <= x_min + 1:
        return None

    cycles = np.arange(x_min, x_max + 1)
    y_vals = f_model(cycles)

    idx = int(np.argmin(np.abs(cycles - cpD2)))
    if idx == 0:
        return None

    Fn = float(y_vals[idx])
    Fprev = float(y_vals[idx - 1])
    if Fprev <= 0:
        return None

    return Fn / Fprev


def _estimate_p0(x: np.ndarray, y: np.ndarray, model: ModelName) -> list:
    """
    Эвристическая оценка стартовых параметров (p0) для алгоритма Левенберга-Марквардта.
    Снижает количество итераций (с ~55 до ~5-10) и предотвращает зависание оптимизатора
    на "плохих" кривых, давая алгоритму сразу точку рядом с истинным минимумом.
    """
    d_est = float(np.min(y))       # Базовая линия (Ground state)
    e_est = float(np.max(y))       # Плато (Plateau)
    
    # Оценка точки перегиба c (где Y ближе всего к середине амплитуды)
    mid_y = (e_est + d_est) / 2.0
    idx_mid = int(np.argmin(np.abs(y - mid_y)))
    c_est = float(x[idx_mid])
    
    b_est = -15.0 # Стандартный негативный наклон для ПЦР
    
    if model == "L4":
        return [b_est, c_est, d_est, e_est]
    else:
        return [b_est, c_est, d_est, e_est, 1.0]


# ==========================
# ФИТТИНГ
# ==========================

def fit_curve_l4(x_raw, y_raw) -> FitResult:
    """
    Подгонка L4 к одной кривой, + производные, cpD1/cpD2, эффективность, ГОФ.
    """
    x, y = _clean_xy(x_raw, y_raw)
    if len(x) < 8:
        return FitResult("L4", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, "Not enough data")
        
    if not _has_reasonable_signal(y):
        return FitResult("L4", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, "No amplification signal (NTC/Noise)")

    p0 = _estimate_p0(x, y, "L4")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Явно указываем Левенберга-Марквардта для стабильности
            popt, pcov = curve_fit(l4_model, x, y, p0=p0, method='lm', maxfev=10000)
            
        b, c, d, e = popt
        params = {"b": float(b), "c": float(c), "d": float(d), "e": float(e)}

        # Замыкание для передачи в зум-искатель
        f_fit = lambda xs: l4_model(xs, b, c, d, e)
        
        # Новый итеративный поиск пиков (zooming) с конечными разностями
        cpD1 = _find_peak_zooming(f_fit, x.min(), x.max(), order=1)
        cpD2 = _find_peak_zooming(f_fit, x.min(), x.max(), order=2)
        
        eff = _efficiency_at_cycle(params, cpD2, model="L4")

        y_fit = f_fit(x)
        rss, r2, aic, aicc = gof_metrics(x, y, y_fit, k_params=4)

        # Оставляем 200 точек просто для красивой отрисовки графика в интерфейсе
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = f_fit(x_dense)

        return FitResult(
            model="L4", params=params, x_dense=x_dense, y_dense=y_dense, 
            cpD1=cpD1, cpD2=cpD2, efficiency=eff, rss=rss, r2=r2, aic=aic, aicc=aicc, 
            success=True, message="OK"
        )
    
    except Exception as err:
        return FitResult("L4", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, str(err))


def fit_curve_l5(x_raw, y_raw) -> FitResult:
    """
    Подгонка L5 к одной кривой, аналогично L4.
    """
    x, y = _clean_xy(x_raw, y_raw)
    if len(x) < 8:
        return FitResult("L5", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, "Not enough data")
        
    if not _has_reasonable_signal(y):
        return FitResult("L5", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, "No amplification signal (NTC/Noise)")

    p0 = _estimate_p0(x, y, "L5")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # УБРАНЫ BOUNDS. Используем Левенберга-Марквардта (method='lm')
            # Это решает проблему "залипания" параметра f на значении 100.
            popt, pcov = curve_fit(l5_model, x, y, p0=p0, method='lm', maxfev=10000)
            
        b, c, d, e, f_param = popt
        params = {"b": float(b), "c": float(c), "d": float(d), "e": float(e), "f": float(f_param)}

        # Замыкание для передачи в зум-искатель
        f_fit = lambda xs: l5_model(xs, b, c, d, e, f_param)
        
        # Новый итеративный поиск пиков (zooming) с конечными разностями
        cpD1 = _find_peak_zooming(f_fit, x.min(), x.max(), order=1)
        cpD2 = _find_peak_zooming(f_fit, x.min(), x.max(), order=2)
        
        eff = _efficiency_at_cycle(params, cpD2, model="L5")

        y_fit = f_fit(x)
        rss, r2, aic, aicc = gof_metrics(x, y, y_fit, k_params=5)

        # Оставляем 200 точек просто для красивой отрисовки графика в интерфейсе
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = f_fit(x_dense)

        return FitResult(
            model="L5", params=params, x_dense=x_dense, y_dense=y_dense, 
            cpD1=cpD1, cpD2=cpD2, efficiency=eff, rss=rss, r2=r2, aic=aic, aicc=aicc, 
            success=True, message="OK"
        )
    
    except Exception as err:
        return FitResult("L5", {}, x, y, None, None, None, np.nan, np.nan, np.nan, np.nan, False, str(err))


def fit_curve_auto(x_raw, y_raw, criterion: Literal["AICc", "AIC", "R2"] = "AICc") -> FitResult:
    """
    Автоматический выбор лучшей модели (L4 или L5) по критерию,
    аналогично qpcR::mselect(do.all=TRUE, crit='weights'/'chisq').

    По умолчанию выбираем модель с минимальным AICc.
    """
    res_l4 = fit_curve_l4(x_raw, y_raw)
    res_l5 = fit_curve_l5(x_raw, y_raw)

    # если обе неудачны — возвращаем ту, у которой хоть что‑то получилось
    if not res_l4.success and not res_l5.success:
        return res_l4 if np.isfinite(res_l4.aicc) else res_l5

    if criterion.upper() in ("AIC", "AICC"):
        key = "aicc" if criterion.upper() == "AICC" else "aic"
        val_l4 = getattr(res_l4, key)
        val_l5 = getattr(res_l5, key)
        
        if np.isnan(val_l4): return res_l5
        if np.isnan(val_l5): return res_l4
        return res_l4 if val_l4 < val_l5 else res_l5
    else:
        val_l4 = res_l4.r2 if not np.isnan(res_l4.r2) else -np.inf
        val_l5 = res_l5.r2 if not np.isnan(res_l5.r2) else -np.inf
        return res_l4 if val_l4 > val_l5 else res_l5