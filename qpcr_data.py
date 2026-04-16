"""
qpcr_data.py
Модуль работы с входными qPCR-данными
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
import io

BaselineMode = Literal["none", "mean", "median", "lin", "quad"]

@dataclass
class QPCRDataset:
    """Обёртка над DataFrame с мета-информацией"""
    df: pd.DataFrame
    cycle_col: str
    sample_cols: List[str]


def load_qpcr_csv(file_path):
    """
    Загружает CSV файл. Возвращает сырой DataFrame БЕЗ трансформаций.
    """
    if isinstance(file_path, str):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        file_path.seek(0)
        content = file_path.read().decode('utf-8')
    
    lines = content.split('\n')
    
    # Проверяем QIAGEN формат
    is_qiagen = any('QIAGEN' in line or line.startswith('"ID"') for line in lines[:30])
    
    if is_qiagen:
        # Ищем заголовок таблицы
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"'):
                start_idx = i
                break
        
        if start_idx is None:
            raise ValueError("Не найдена таблица данных в QIAGEN файле")
        
        # Обрабатываем и читаем
        processed_lines = []
        for line in lines[start_idx:]:
            processed = line.replace('"', '').replace(',', '.')
            processed_lines.append(processed)
        
        df = pd.read_csv(io.StringIO('\n'.join(processed_lines)), sep=';')
        
        # Транспонируем: sample в строках, циклы в столбцах
        sample_names = df.iloc[0, 1:].tolist()
        df_data = df.iloc[1:].reset_index(drop=True)
        
        # Фильтруем только целые циклы (1-100)
        first_col = df_data.iloc[:, 0].astype(str).str.strip()
        valid_rows = []
        for idx, val in enumerate(first_col):
            try:
                cycle_num = int(float(val))
                if 1 <= cycle_num <= 100:
                    valid_rows.append(idx)
            except (ValueError, TypeError):
                continue
        
        if not valid_rows:
            raise ValueError("Не найдены циклы амплификации (1-100)")
        
        df_data = df_data.iloc[valid_rows].reset_index(drop=True)
        cycles = [int(float(df_data.iloc[i, 0])) for i in range(len(df_data))]
        
        # Транспонируем: sample → строки, cycle_X → столбцы
        df_final = df_data.iloc[:, 1:].T
        df_final.columns = [f'cycle_{c}' for c in cycles]
        df_final.insert(0, 'sample', sample_names)
        df_final = df_final.reset_index(drop=True)
        
        return df_final
    
    else:
        # Обычный формат
        file_path.seek(0) if hasattr(file_path, 'seek') else None
        for sep in [',', ';', '\t']:
            try:
                if isinstance(file_path, str):
                    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                else:
                    file_path.seek(0)
                    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        raise ValueError("Не удалось загрузить файл")


def convert_qiagen_to_normal(file_or_df):
    """
    Конвертирует QIAGEN формат в нормальный.
    """
    import io
    
    # Читаем файл заново
    if isinstance(file_or_df, str):
        with open(file_or_df, 'r', encoding='utf-8') as f:
            content = f.read()
    elif hasattr(file_or_df, 'read'):
        file_or_df.seek(0)
        content = file_or_df.read().decode('utf-8')
    else:
        raise ValueError("Передайте файл, а не DataFrame")
    
    lines = content.split('\n')
    
    # Ищем заголовок
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError("Не найдена таблица данных")
    
    # Обрабатываем строки
    processed_lines = []
    for line in lines[start_idx:]:
        processed = line.replace('"', '').replace(',', '.')
        processed_lines.append(processed)
    
    df = pd.read_csv(io.StringIO('\n'.join(processed_lines)), sep=';')
    
    # Извлекаем имена образцов из первой строки
    sample_names = df.iloc[0, 1:].tolist()
    
    # Данные со второй строки
    df_data = df.iloc[1:].reset_index(drop=True)
    first_col = df_data.iloc[:, 0].astype(str).str.strip()
    
    # Фильтруем только целые циклы 1-100
    valid_rows = []
    cycles = []
    for idx, val in enumerate(first_col):
        try:
            cycle_num = int(float(val))
            if 1 <= cycle_num <= 100:
                valid_rows.append(idx)
                cycles.append(cycle_num)
        except (ValueError, TypeError):
            continue
    
    if not valid_rows:
        raise ValueError("Не найдены циклы амплификации")
    
    # Берём только валидные строки
    df_data = df_data.iloc[valid_rows].reset_index(drop=True)
    
    # Извлекаем данные (БЕЗ первого столбца ID)
    data_values = df_data.iloc[:, 1:].values  # numpy array, строки=циклы, столбцы=образцы
    
    # Создаём DataFrame
    df_final = pd.DataFrame(data_values, columns=sample_names)
    df_final.insert(0, 'Cycle', cycles)
    
    # Преобразуем типы
    for col in sample_names:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    return df_final



def detect_cycle_column(df: pd.DataFrame) -> str:
    """Эвристика для поиска колонки циклов"""
    candidates = ["cycle", "cycles", "cyc", "ct", "cq"]
    for col in df.columns:
        if any(key in col.lower() for key in candidates):
            return col
    return df.columns[0]


def coerce_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """Приводим колонки к числовому виду"""
    if exclude is None:
        exclude = []
    df_num = df.copy()
    for col in df_num.columns:
        if col in exclude:
            continue
        s = df_num[col].astype(str)
        s = s.str.replace(",", ".", regex=False)
        df_num[col] = pd.to_numeric(s, errors="coerce")
    return df_num


def select_sample_columns(df: pd.DataFrame, cycle_col: str, min_non_na: int = 3) -> List[str]:
    """Определяет числовые колонки для проб (Y)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_cols = [c for c in numeric_cols if c != cycle_col]
    good_cols: List[str] = []
    for col in sample_cols:
        non_na = df[col].notna().sum()
        if non_na >= min_non_na:
            good_cols.append(col)
    return good_cols


def _baseline_range_mask(df: pd.DataFrame, cycle_col: str, start_cycle: float, end_cycle: float) -> np.ndarray:
    """Маска строк baseline"""
    if cycle_col not in df.columns:
        raise ValueError(f"Колонка циклов '{cycle_col}' не найдена")
    x = pd.to_numeric(df[cycle_col], errors="coerce")
    mask = (x >= start_cycle) & (x <= end_cycle)
    return mask.values


def baseline_subtract(dataset: QPCRDataset, start_cycle: float, end_cycle: float,
                     mode: BaselineMode = "mean", base_factor: float = 1.0) -> QPCRDataset:
    """Вычитание baseline"""
    if mode == "none":
        return dataset
    
    df = dataset.df.copy()
    x_col = dataset.cycle_col
    y_cols = dataset.sample_cols
    mask = _baseline_range_mask(df, x_col, start_cycle, end_cycle)
    
    if not mask.any():
        raise ValueError(f"Диапазон [{start_cycle}, {end_cycle}] не содержится в данных")
    
    x_base = pd.to_numeric(df.loc[mask, x_col], errors="coerce").values
    
    for col in y_cols:
        y = pd.to_numeric(df[col], errors="coerce").values
        y_base = y[mask]
        
        if np.all(~np.isfinite(y_base)):
            continue
        
        if mode == "mean":
            baseline = np.nanmean(y_base) * base_factor
            df[col] = y - baseline
        elif mode == "median":
            baseline = np.nanmedian(y_base) * base_factor
            df[col] = y - baseline
        elif mode in ("lin", "quad"):
            x_valid = x_base[np.isfinite(y_base)]
            y_valid = y_base[np.isfinite(y_base)]
            if len(x_valid) < 2:
                continue
            X = np.vstack([np.ones_like(x_valid), x_valid]).T
            if mode == "quad":
                X = np.vstack([np.ones_like(x_valid), x_valid, x_valid**2]).T
            beta, *_ = np.linalg.lstsq(X, y_valid, rcond=None)
            x_all = pd.to_numeric(df[x_col], errors="coerce").values
            if mode == "lin":
                y_hat = beta[0] + beta[1] * x_all
            else:
                y_hat = beta[0] + beta[1] * x_all + beta[2] * (x_all**2)
            baseline = y_hat * base_factor
            df[col] = y - baseline
        else:
            raise ValueError(f"Неизвестный режим baseline: {mode}")
        
        df[col] = df[col].apply(lambda v: max(v, 0.001) if pd.notna(v) else v)
    
    return QPCRDataset(df=df, cycle_col=x_col, sample_cols=list(y_cols))


def build_dataset_from_raw(df_raw: pd.DataFrame, cycle_col: Optional[str] = None,
                           min_non_na: int = 3) -> QPCRDataset:
    """Конструктор QPCRDataset из сырых данных"""
    if cycle_col is None:
        cycle_col = detect_cycle_column(df_raw)
    df_num = coerce_numeric_columns(df_raw, exclude=[cycle_col])
    sample_cols = select_sample_columns(df_num, cycle_col, min_non_na=min_non_na)
    if not sample_cols:
        raise ValueError("Не удалось найти числовые колонки для проб (Y)")
    return QPCRDataset(df=df_num, cycle_col=cycle_col, sample_cols=sample_cols)

