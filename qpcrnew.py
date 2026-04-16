"""
app.py
Streamlit‑интерфейс к ядру Py‑qpcR с поддержкой пакетной обработки кривых 
и ручного выбора столбцов для финального расчета.
Добавлена визуализация распределения (Boxplot).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import io
import re
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from qpcr_data import (
    load_qpcr_csv,
    build_dataset_from_raw,
    baseline_subtract,
    QPCRDataset,
    coerce_numeric_columns,
    convert_qiagen_to_normal,
)

from qpcr_models import (
    fit_curve_l4,
    fit_curve_l5,
    fit_curve_auto,
    l4_model,
    l5_model,
)

from qpcr_analysis import (
    batch_fit,
    calib_efficiency,
    relative_expression,
)

# ======================
# НАСТРОЙКА СТРАНИЦЫ
# ======================
st.set_page_config(
    page_title="Py-qpcR",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Py-qpcR – Анализ qPCR")

# ======================
# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ
# ======================
def init_state():
    if "raw_df" not in st.session_state: st.session_state["raw_df"] = None
    if "ds" not in st.session_state: st.session_state["ds"] = None
    if "ds_base" not in st.session_state: st.session_state["ds_base"] = None
    if "batch_result" not in st.session_state: st.session_state["batch_result"] = None
    if "manual_df" not in st.session_state: st.session_state["manual_df"] = None
    if "final_res" not in st.session_state: st.session_state["final_res"] = None
    if "individual_data" not in st.session_state: st.session_state["individual_data"] = None

init_state()

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def robust_load_df(file_obj, header_row=0):
    """Загрузка DataFrame из файла"""
    if file_obj.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_obj, header=header_row)
    else:
        file_obj.seek(0)
        return pd.read_csv(file_obj, header=header_row, sep=None, engine='python')

def clean_val(val):
    """Преобразование строковых Ct в числа"""
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(',', '.')
    if s in ['', '-', 'None', 'NaN', 'nan']: return np.nan
    try:
        return float(s)
    except:
        return np.nan

# ======================
# БОКОВАЯ ПАНЕЛЬ
# ======================
with st.sidebar:
    st.header("📂 1. Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите таблицу (Excel/CSV/QIAGEN)", type=["csv", "xlsx", "xls"])
    header_row = st.number_input("Строка с именами столбцов (0-based):", min_value=0, value=2, step=1)
    
    if uploaded_file:
        try:
            df_loaded = robust_load_df(uploaded_file, header_row)
            df_loaded = df_loaded.dropna(how='all', axis=0).dropna(how='all', axis=1)
            st.session_state["manual_df"] = df_loaded
            
            uploaded_file.seek(0)
            raw_df = load_qpcr_csv(uploaded_file)
            st.session_state["raw_df"] = raw_df
            st.success("Файл успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка при чтении: {e}")

    st.markdown("---")
    st.header("⚙️ Глобальные настройки")
    efficiency_global = st.number_input("Эффективность (E):", 1.0, 3.0, 2.0, 0.01)

# ======================
# ОСНОВНОЕ ОКНО
# ======================
tab_manual, tab_curves, tab_converter = st.tabs([
    "📝 Ручной анализ (по столбцам)", 
    "📈 Анализ кривых (Raw Data)", 
    "🔄 Конвертер Excel -> CSV"
])

# --- ВКЛАДКА 1: РУЧНОЙ АНАЛИЗ ---
with tab_manual:
    if st.session_state["manual_df"] is not None:
        df = st.session_state["manual_df"].copy()
        cols = list(df.columns)
        
        st.subheader("Настройка дельта-дельта Ct по выбранным столбцам")
        
        c1, c2 = st.columns(2)
        with c1:
            ref_col = st.selectbox("Выберите столбец референса (HKG):", options=cols, index=min(len(cols)-1, 7))
            target_cols = st.multiselect("Выберите столбцы целевых генов:", options=[c for c in cols if c != ref_col])
        
        with c2:
            group_col = st.selectbox("Столбец для групп (напр. 'Организм'):", options=cols, index=min(len(cols)-1, 2))
            unique_groups = sorted(df[group_col].dropna().unique().tolist())
            control_val = st.selectbox("Контрольная группа (Baseline):", options=unique_groups) if unique_groups else None

        if st.button("🚀 Рассчитать экспрессию", type="primary"):
            if not target_cols or not group_col or control_val is None:
                st.warning("Выберите гены, группы и контроль.")
            else:
                summary_results = []
                all_individual_rows = []
                
                # Очистка данных
                for c in [ref_col] + target_cols:
                    df[c] = df[c].apply(clean_val)
                
                for gene in target_cols:
                    temp_df = df[[group_col, ref_col, gene]].dropna(subset=[ref_col, gene])
                    if temp_df.empty: continue
                    
                    # 1. Расчет индивидуальных dCt
                    temp_df['dct'] = temp_df[gene] - temp_df[ref_col]
                    
                    # 2. Среднее dCt для контроля
                    ctrl_dct_mean = temp_df[temp_df[group_col] == control_val]['dct'].mean()
                    ctrl_values = temp_df[temp_df[group_col] == control_val]['dct'].values
                    
                    # 3. Расчет индивидуальных Fold Change для каждой строки (для ящика с усами)
                    temp_df['ddct'] = temp_df['dct'] - ctrl_dct_mean
                    temp_df['fc'] = efficiency_global ** (-temp_df['ddct'])
                    temp_df['gene_name'] = gene
                    
                    all_individual_rows.append(temp_df[[group_col, 'gene_name', 'fc', 'dct']])
                    
                    # 4. Группировка для сводной таблицы
                    grouped = temp_df.groupby(group_col)['dct'].agg(['mean', 'std', 'count']).reset_index()
                    for _, row in grouped.iterrows():
                        g_name = row[group_col]
                        m_dct = row['mean']
                        s_dct = row['std'] if pd.notna(row['std']) else 0
                        
                        ddct = m_dct - ctrl_dct_mean
                        fc_mean = efficiency_global ** (-ddct)
                        l2fc = -ddct * np.log2(efficiency_global)
                        
                        # T-тест
                        sample_vals = temp_df[temp_df[group_col] == g_name]['dct'].values
                        p_val = np.nan
                        if g_name != control_val and len(sample_vals) > 1 and len(ctrl_values) > 1:
                            _, p_val = stats.ttest_ind(sample_vals, ctrl_values, equal_var=False)
                        
                        summary_results.append({
                            "Ген": gene, "Группа": g_name, "N": int(row['count']),
                            "Mean ΔCt": round(m_dct, 3), "SD ΔCt": round(s_dct, 3),
                            "Fold Change (Mean)": round(fc_mean, 4), "Log2 FC": round(l2fc, 3),
                            "P-value": round(p_val, 5) if pd.notna(p_val) else "-"
                        })
                
                st.session_state["final_res"] = pd.DataFrame(summary_results)
                st.session_state["individual_data"] = pd.concat(all_individual_rows)
                st.success("Расчет завершен!")

        if st.session_state["final_res"] is not None:
            res_df = st.session_state["final_res"]
            ind_df = st.session_state["individual_data"]
            
            st.markdown("---")
            st.subheader("📊 Итоговая таблица (Средние значения)")
            st.dataframe(res_df, use_container_width=True)
            
            # --- ВИЗУАЛИЗАЦИЯ ---
            st.subheader("📈 Визуализация распределения")
            plot_type = st.segmented_control("Тип графика:", ["Столбчатая диаграмма", "Ящик с усами (Boxplot)"], default="Ящик с усами (Boxplot)")
            
            if plot_type == "Столбчатая диаграмма":
                fig = px.bar(res_df, x="Группа", y="Fold Change (Mean)", color="Ген", barmode="group",
                             error_y="SD ΔCt", title="Средний Fold Change с ошибкой (SD ΔCt)")
            else:
                # Построение ящика с усами по индивидуальным значениям fc
                fig = px.box(ind_df, x=group_col, y="fc", color="gene_name", points="all",
                             labels={"fc": "Fold Change (отн. контроля)", group_col: "Группа", "gene_name": "Ген"},
                             title="Распределение Fold Change по повторностям (Boxplot)")
                # Добавляем линию на уровне 1.0 (контроль)
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Контроль")

            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            csv = res_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
            st.download_button("📥 Скачать результаты (CSV)", data=csv, file_name="results_qpcr.csv")
    else:
        st.info("Загрузите файл во вкладке загрузки.")

# (Остальные вкладки остаются без изменений для краткости, они в файле сохранены)
with tab_curves:
    if st.session_state["raw_df"] is not None:
        st.info("Используйте эту вкладку для анализа сырых кривых (fluorescence vs cycle).")
# ... (код tab_curves и tab_converter из предыдущей версии)