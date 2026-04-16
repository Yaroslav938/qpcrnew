"""
qpcr_experiment.py

Высокоуровневый анализ экспериментов по экспрессии:
- Группировка технических повторов и усреднение Ct
- Расчёт ΔCt (нормализация на референс)
- Регрессионная нормализация (на основе Wang et al., 2015)
- Расчёт ΔΔCt и Fold Change (с учетом эффективности — метод Пфаффла)
- Статистика (t-test между группами)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import re


@dataclass
class GroupedSample:
    """
    Результат группировки реплик для одного биологического образца/гена.
    
    sample_name : имя образца (без суффиксов _Rep1, _Rep2 и т.п.)
    ct_mean     : среднее Ct по репликам
    ct_sd       : стандартное отклонение Ct
    eff_mean    : средняя эффективность
    n_replicates: количество реплик
    """
    sample_name: str
    ct_mean: float
    ct_sd: float
    eff_mean: float
    n_replicates: int


def group_replicates(batch_table: pd.DataFrame, sample_col: str = "sample",
                   ct_col: str = "Ct_cpD2", eff_col: str = "Efficiency_cpD2",
                   group_pattern: Optional[str] = None) -> pd.DataFrame:
    
    df = batch_table.copy()
    # Очистка от невидимых пробелов по краям
    df[sample_col] = df[sample_col].astype(str).str.strip()
    
    # 1. Пробуем применить регулярное выражение из Streamlit
    if group_pattern and group_pattern.strip():
        try:
            # str.extract вытаскивает ту часть имени, которая совпала с шаблоном (...)
            extracted = df[sample_col].str.extract(group_pattern, expand=False)
            if isinstance(extracted, pd.DataFrame):
                extracted = extracted.iloc[:, 0]
            # Если регулярка не сработала, оставляем оригинальное имя
            df["group_name"] = extracted.fillna(df[sample_col])
        except Exception as e:
            print(f"Ошибка регулярного выражения: {e}")
            df["group_name"] = df[sample_col]
    else:
        df["group_name"] = df[sample_col]
    
    # 2. АВТО-ГРУППИРОВКА (Умный fallback)
    # Если регулярка не отработала (названия остались старыми), удаляем хвосты принудительно
    if (df["group_name"] == df[sample_col]).all():
        df["group_name"] = df["group_name"].str.replace(r'\s*\[\d+\]$|\s+\d+$|_\d+$|\.\d+$', '', regex=True)
    
    # Группируем
    grouped = df.groupby("group_name").agg({
        ct_col: ['mean', 'std'],
        eff_col: 'mean',
        sample_col: 'count'
    }).round(3)
    
    grouped.columns = ['ct_mean', 'ct_sd', 'eff_mean', 'n_replicates']
    grouped = grouped.reset_index()
    grouped['ct_sd'] = grouped['ct_sd'].fillna(0)
    
    print(f"✅ Сгруппировано: {len(grouped)} образцов")
    return grouped


@dataclass
class DeltaCtResult:
    """
    Результат расчёта ΔCt (нормализация на референс).
    
    sample      : имя образца (target)
    ct_target   : среднее Ct target-гена
    ct_ref      : среднее Ct референсного гена
    delta_ct    : ΔCt = ct_target - ct_ref
    """
    sample: str
    ct_target: float
    ct_ref: float
    delta_ct: float


def calculate_delta_ct(
    grouped_table: pd.DataFrame,
    target_samples: List[str],
    reference_samples: List[str],
) -> List[DeltaCtResult]:
    """
    Рассчитывает ΔCt для каждого target-образца относительно усреднённого референса.
    
    Параметры:
        grouped_table     : таблица после group_replicates
        target_samples    : список имён target-генов/образцов
        reference_samples : список имён референсных генов (будут усреднены геометрически)
    
    Возвращает:
        Список DeltaCtResult
    """
    # Усредняем референсные гены (геометрическое среднее Ct)
    ref_rows = grouped_table[grouped_table["group_name"].isin(reference_samples)]
    if ref_rows.empty:
        raise ValueError("Не найдено ни одного референсного образца в таблице.")
    
    ct_ref_geomean = float(ref_rows["ct_mean"].mean())  # можно заменить на geomean если нужно
    
    results = []
    for target in target_samples:
        target_row = grouped_table[grouped_table["group_name"] == target]
        if target_row.empty:
            continue
        ct_target = float(target_row["ct_mean"].iloc[0])
        delta_ct = ct_target - ct_ref_geomean
        results.append(
            DeltaCtResult(
                sample=target,
                ct_target=ct_target,
                ct_ref=ct_ref_geomean,
                delta_ct=delta_ct,
            )
        )
    
    return results


@dataclass
class FoldChangeResult:
    """
    Результат расчёта Fold Change (метод ΔΔCt или Пфаффла).
    
    sample          : имя образца
    delta_ct        : ΔCt образца
    delta_delta_ct  : ΔΔCt = delta_ct - mean(delta_ct_control)
    fold_change     : 2^(-ΔΔCt) или по Пфаффлу с учетом E
    log2_fc         : log2(fold_change)
    """
    sample: str
    delta_ct: float
    delta_delta_ct: float
    fold_change: float
    log2_fc: float


def calculate_fold_change(
    delta_ct_list: List[DeltaCtResult],
    control_samples: List[str],
    method: str = "standard",
    eff_target: float = 2.0,
    eff_ref: float = 2.0,
) -> List[FoldChangeResult]:
    """
    Рассчитывает Fold Change относительно контрольной группы.
    
    Параметры:
        delta_ct_list    : список DeltaCtResult
        control_samples  : имена образцов контрольной группы
        method           : "standard" (2^-ΔΔCt) или "pfaffl" (учитывает эффективность)
        eff_target       : средняя эффективность target-гена (для Пфаффла)
        eff_ref          : средняя эффективность референса (для Пфаффла)
    
    Возвращает:
        Список FoldChangeResult
    """
    # Среднее ΔCt контрольной группы
    control_dcts = [r.delta_ct for r in delta_ct_list if r.sample in control_samples]
    if not control_dcts:
        raise ValueError("Не найдено контрольных образцов в delta_ct_list.")
    
    mean_control_dct = float(np.mean(control_dcts))
    
    results = []
    for dct_res in delta_ct_list:
        ddct = dct_res.delta_ct - mean_control_dct
        
        if method == "standard":
            fc = 2.0 ** (-ddct)
        elif method == "pfaffl":
            # Метод Пфаффла: (E_target^-ΔCt_target) / (E_ref^-ΔCt_ref)
            fc = eff_target ** (-ddct)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        log2_fc = float(np.log2(fc)) if fc > 0 else np.nan
        
        results.append(
            FoldChangeResult(
                sample=dct_res.sample,
                delta_ct=dct_res.delta_ct,
                delta_delta_ct=ddct,
                fold_change=fc,
                log2_fc=log2_fc,
            )
        )
    
    return results


@dataclass
class StatTestResult:
    """
    Результат статистического сравнения двух групп.
    
    group1      : имя первой группы
    group2      : имя второй группы
    t_statistic : t-статистика
    p_value     : p-value
    mean_diff   : разница средних ΔCt
    """
    group1: str
    group2: str
    t_statistic: float
    p_value: float
    mean_diff: float


def compare_groups_ttest(
    raw_table: pd.DataFrame,
    group1_samples: List[str],
    group2_samples: List[str],
    value_col: str = "delta_ct",
) -> StatTestResult:
    """
    T-тест между двумя группами по значениям ΔCt (или другой метрике).
    
    Параметры:
        raw_table       : таблица с исходными данными (до усреднения или после)
        group1_samples  : список имён образцов группы 1
        group2_samples  : список имён образцов группы 2
        value_col       : имя колонки со значениями для сравнения
    
    Возвращает:
        StatTestResult
    """
    g1_values = raw_table[raw_table["sample"].isin(group1_samples)][value_col].dropna()
    g2_values = raw_table[raw_table["sample"].isin(group2_samples)][value_col].dropna()
    
    if len(g1_values) < 2 or len(g2_values) < 2:
        raise ValueError("Недостаточно данных для t-теста (нужно минимум 2 значения в каждой группе).")
    
    t_stat, p_val = stats.ttest_ind(g1_values, g2_values)
    mean_diff = float(g1_values.mean() - g2_values.mean())
    
    return StatTestResult(
        group1="Group1",
        group2="Group2",
        t_statistic=float(t_stat),
        p_value=float(p_val),
        mean_diff=mean_diff,
    )


def parse_sample_structure(sample_name: str) -> Tuple[str, str]:
    """
    Умный парсинг структуры вида: 'ген группа [номер]'
    Поддерживает пробелы в начале, точки в названии гена (gsta.1) и двойные пробелы.
    """
    # 1. Очищаем пробелы и кавычки по краям
    sample_name = sample_name.strip().strip('"').strip()
    
    # 2. Удаляем технические идентификаторы реплик: [1], _1, .1
    cleaned = re.sub(r'\s*\[\d+\]$|\s+\d+$|_\d+$|\.\d+$', '', sample_name)
    
    # 3. Разбиваем по пробелам. 
    # Первое слово всегда считаем геном, всё остальное - биологической группой.
    parts = cleaned.split()
    
    if len(parts) >= 2:
        gene = parts[0].strip()
        bio_group = " ".join(parts[1:]).strip()
        return gene, bio_group
    elif len(parts) == 1:
        return parts[0].strip(), "Unknown"
    
    return "Unknown", "Unknown"


def automated_experiment_analysis(
    grouped_table: pd.DataFrame,
    raw_table: pd.DataFrame,
    reference_genes: List[str],
    control_group: str,
    use_regression_norm: bool = True,  # Использовать ли метод Wang et al. 2015
    alpha_reg: float = 0.05            # Порог значимости для коэффициента b
) -> pd.DataFrame:
    """
    Автоматический анализ всего эксперимента с расчётом p-value на исходных повторах.
    Реализует регрессионный метод нормализации с модификацией пользователя:
    ΔCt = b * Ct_control - Ct_target.
    """
    # Парсим структуру
    grouped_table = grouped_table.copy()
    grouped_table["gene"] = grouped_table["group_name"].apply(lambda x: parse_sample_structure(x)[0])
    grouped_table["bio_group"] = grouped_table["group_name"].apply(lambda x: parse_sample_structure(x)[1])
    
    raw_table = raw_table.copy()
    raw_table["gene"] = raw_table["sample"].apply(lambda x: parse_sample_structure(x)[0])
    
    # Для исходной таблицы bio_group вычисляем тем же новым парсером
    raw_table["bio_group"] = raw_table["sample"].apply(lambda x: parse_sample_structure(x)[1])
    
    ref_table = grouped_table[grouped_table["gene"].isin(reference_genes)]
    target_table = grouped_table[~grouped_table["gene"].isin(reference_genes)]
    
    if ref_table.empty:
        raise ValueError("Референсные гены не найдены в данных. Проверьте правильность написания.")
    
    # Среднее Ct референсов по группам
    ref_ct_by_group = ref_table.groupby("bio_group")["ct_mean"].mean().to_dict()
    
    # Референсные Ct для исходных повторов
    raw_ref_table = raw_table[raw_table["gene"].isin(reference_genes)]
    raw_ref_ct_by_group = raw_ref_table.groupby("bio_group")["Ct_cpD2"].mean().to_dict()
    
    # Рассчитываем усреднённые ΔCt для target-генов с применением регрессии
    results = []
    raw_delta_ct_list = []
    
    for gene in target_table["gene"].unique():
        gene_data = target_table[target_table["gene"] == gene]
        
        # Списки для построения линейной регрессии
        x_vals = [] # Ct Control (Reference)
        y_vals = [] # Ct Target
        bio_groups_valid = []
        
        for _, row in gene_data.iterrows():
            bio_group = row["bio_group"]
            ct_target = row["ct_mean"]
            ct_ref = ref_ct_by_group.get(bio_group, np.nan)
            
            if pd.notna(ct_ref) and pd.notna(ct_target):
                x_vals.append(ct_ref)
                y_vals.append(ct_target)
                bio_groups_valid.append(bio_group)
                
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        b_coeff = 1.0  # Коэффициент по умолчанию (классическое вычитание)
        
        if use_regression_norm and len(x_vals) >= 3:
            # 1) Удаление выбросов в референсе (IQR метод с подстраховкой)
            q1 = np.percentile(x_vals, 25)
            q3 = np.percentile(x_vals, 75)
            iqr = max(q3 - q1, 0.5)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (x_vals >= lower_bound) & (x_vals <= upper_bound)
            x_filt = x_vals[mask]
            y_filt = y_vals[mask]
            
            # 2) Линейная регрессия Target (Y) на Control (X)
            if len(x_filt) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_filt, y_filt)
                # 3) Если коэффициент значим, используем его
                if p_value < alpha_reg:
                    b_coeff = slope
                    
        # Вычисление ΔCt по формуле: b * Ct_control - Ct_target
        for bio_group, x_val, y_val in zip(bio_groups_valid, x_vals, y_vals):
            delta_ct = (b_coeff * x_val) - y_val
            results.append({
                "gene": gene,
                "bio_group": bio_group,
                "delta_ct": delta_ct,
                "b_coeff": b_coeff
            })
            
        # Для сырых повторов (нужно для честного расчета p-value t-теста)
        raw_target_table = raw_table[~raw_table["gene"].isin(reference_genes)]
        gene_raw_data = raw_target_table[raw_target_table["gene"] == gene]
        
        for _, row in gene_raw_data.iterrows():
            bio_group = row["bio_group"]
            ct_target_raw = row["Ct_cpD2"]
            ct_ref_raw = raw_ref_ct_by_group.get(bio_group, np.nan)
            
            if pd.notna(ct_target_raw) and pd.notna(ct_ref_raw):
                delta_ct_raw = (b_coeff * ct_ref_raw) - ct_target_raw
                raw_delta_ct_list.append({
                    "gene": gene,
                    "bio_group": bio_group,
                    "delta_ct_raw": delta_ct_raw,
                })
                
    delta_ct_df = pd.DataFrame(results)
    raw_delta_ct_df = pd.DataFrame(raw_delta_ct_list)
    
    if delta_ct_df.empty:
        raise ValueError("Не удалось рассчитать ΔCt. Возможно, имена групп в референсных и целевых генах не совпадают.")
        
    # Итоговая таблица с Fold Change и p-value для target-генов
    final_results = []
    
    for gene in delta_ct_df["gene"].unique():
        gene_data = delta_ct_df[delta_ct_df["gene"] == gene]
        gene_raw_data = raw_delta_ct_df[raw_delta_ct_df["gene"] == gene]
        
        # Защита: ищем частичное совпадение контрольной группы, если точного нет
        control_data = gene_data[gene_data["bio_group"].str.contains(control_group, regex=False)]
        
        if control_data.empty:
            # Если контроль совсем не найден для этого гена - пропускаем его
            continue
            
        mean_control_dct = control_data["delta_ct"].mean()
        
        # Ищем контрольную группу в сырых данных для t-test
        control_raw_mask = gene_raw_data["bio_group"].str.contains(control_group, regex=False)
        control_raw = gene_raw_data[control_raw_mask]["delta_ct_raw"]
        
        current_b = gene_data["b_coeff"].iloc[0]
        
        for bio_group in gene_data["bio_group"].unique():
            group_data = gene_data[gene_data["bio_group"] == bio_group]
            mean_dct = group_data["delta_ct"].mean()
            sd_dct = group_data["delta_ct"].std() if len(group_data) > 1 else 0.0
            
            ddct = mean_dct - mean_control_dct
            
            # ВНИМАНИЕ: Формула Fold Change изменена на 2^(+ΔΔCt), так как
            # ΔCt теперь считается как Ref - Target (больше ΔCt = больше экспрессия)
            fold_change = 2.0 ** ddct
            log2_fc = ddct
            
            group_raw = gene_raw_data[gene_raw_data["bio_group"] == bio_group]["delta_ct_raw"]
            
            # Сравниваем только если это не контрольная группа
            if not (control_group in bio_group) and len(control_raw) >= 2 and len(group_raw) >= 2:
                # В случае вырождения (нулевая дисперсия) ttest упадет, ловим это
                try:
                    t_stat, p_val = stats.ttest_ind(control_raw, group_raw)
                except Exception:
                    p_val = np.nan
            else:
                p_val = np.nan
            
            final_results.append({
                "Gene": gene,
                "Group": bio_group,
                "Regr_b": current_b,
                "ΔCt_mean": mean_dct,
                "ΔCt_sd": sd_dct,
                "ΔΔCt": ddct,
                "Fold_Change": fold_change,
                "Log2_FC": log2_fc,
                "P_value": p_val,
            })
            
    # --- Добавляем референсные гены в итоговую таблицу ---
    ref_results = []
    
    for _, row in ref_table.iterrows():
        bio_group = row["bio_group"]
        gene = row["gene"]
        
        ref_results.append({
            "Gene": gene,
            "Group": bio_group,
            "Regr_b": 1.0,
            "ΔCt_mean": 0.0,
            "ΔCt_sd": 0.0,
            "ΔΔCt": 0.0,
            "Fold_Change": 1.0,
            "Log2_FC": 0.0,
            "P_value": np.nan,
        })
        
    # Объединяем target и референсные гены
    final_df = pd.DataFrame(final_results)
    ref_df = pd.DataFrame(ref_results)
    
    if not final_df.empty or not ref_df.empty:
        combined = pd.concat([final_df, ref_df], ignore_index=True)
        # Удаляем дубликаты референсных генов, которые могли возникнуть при слиянии
        combined = combined.drop_duplicates(subset=['Gene', 'Group'])
        return combined
    else:
        return pd.DataFrame()