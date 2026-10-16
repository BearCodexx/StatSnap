import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
import statsmodels.api as sm

from utils.data_processing import (
    load_dataset,
    optimize_dataframe_memory,
    export_dataframe_bytes
)
from knowledge_base import knowledge_base

warnings.filterwarnings('ignore')

# конфигурация страницы
st.set_page_config(
    page_title="StatSnap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# строгие профессиональные стили без иишных градиентов
st.markdown("""
<style>
    /* типографика и глобальный фон */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* минималистичные карточки метрик */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(128, 128, 128, 0.9);
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 12px;
        color: #10b981;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* карточки стартового экрана */
    .hero-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .hero-tag {
        display: inline-block;
        font-family: monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(128, 128, 128, 0.12);
        margin-bottom: 12px;
        width: fit-content;
    }
    .hero-title {
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .hero-desc {
        font-size: 13px;
        color: rgba(128, 128, 128, 0.9);
        line-height: 1.5;
    }
    
    /* баннер стартовой страницы */
    .welcome-panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 24px 30px;
        margin-bottom: 24px;
    }
    
    /* бейджи для технической инфы */
    .badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 11px;
        font-family: monospace;
        background: rgba(128, 128, 128, 0.15);
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# инициализация сессионного состояния
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'mem_stats' not in st.session_state:
    st.session_state.mem_stats = None


# функции генерации аналитических отчетов

def explain_linear_regression(sm_model, y_true, y_pred, feature_names, target_name):
    # текстовая расшифровка линейной регрессии
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    p_val_f = sm_model.f_pvalue

    sig_vars = [var for var, p in zip(sm_model.params.index, sm_model.pvalues) if p < 0.05 and var != 'const']

    lines = [
        f"РЕЗЮМЕ МОДЕЛИ: {target_name}",
        f"--------------------------------------------------",
        f"• Качество подгонки R²: {r2:.4f} (скорректированный: {sm_model.rsquared_adj:.4f})",
        f"• Средняя ошибка RMSE: {rmse:.4f} | MAE: {mae:.4f}",
        f"• Общая значимость F-статистики: p-value = {p_val_f:.4e} ({'модель адекватна' if p_val_f < 0.05 else 'модель статистически не значима'})",
        f"• Значимых факторов (p < 0.05): {len(sig_vars)} из {len(feature_names)} ({', '.join(sig_vars) if sig_vars else 'нет'})",
        "",
        "УРАВНЕНИЕ РЕГРЕССИИ:",
    ]

    intercept = sm_model.params.get('const', 0.0)
    eq_parts = [f"{intercept:.4f}"]
    for col in feature_names:
        if col in sm_model.params:
            coef = sm_model.params[col]
            sign = "+" if coef >= 0 else "-"
            eq_parts.append(f"{sign} {abs(coef):.4f} * {col}")

    lines.append(f"{target_name} = " + " ".join(eq_parts))
    return "\n".join(lines)


# сайдбар: загрузка данных и управление ресурсами
with st.sidebar:
    st.markdown("### Данные и конфигурация")
    uploaded_file = st.file_uploader(
        "Загрузить датасет",
        type=['csv', 'txt', 'xlsx', 'xls'],
        help="Поддерживаются CSV с любыми разделителями и кодировками (UTF-8, CP1251), а также Excel."
    )

    # загрузка выполняется строго единожды при смене файла, предотвращая утечки ram
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.file_hash != file_id:
            try:
                with st.spinner("Загрузка и оптимизация типов..."):
                    df_loaded, mem_info = load_dataset(uploaded_file, optimize=True)
                    st.session_state.df = df_loaded
                    st.session_state.df_original = df_loaded.copy()
                    st.session_state.filename = uploaded_file.name
                    st.session_state.file_hash = file_id
                    st.session_state.mem_stats = mem_info
                    # сбрасываем старый экспортный буфер
                    st.session_state.pop('export_cache', None)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

    # блок информации о датасете в памяти
    if st.session_state.df is not None:
        df_curr = st.session_state.df
        st.markdown("---")
        st.markdown("### Сводка окружения")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Строк", f"{df_curr.shape[0]:,}")
        with c2:
            st.metric("Колонок", df_curr.shape[1])

        mem_mb = df_curr.memory_usage(deep=True).sum() / (1024 * 1024)
        st.markdown(f"**Память в RAM:** `{mem_mb:.2f} MB`")

        if st.session_state.mem_stats and st.session_state.mem_stats.get("saved_pct", 0) > 0:
            saved = st.session_state.mem_stats["saved_pct"]
            st.caption(f"Экономия RAM за счет сжатия типов: `{saved:.1f}%`")

        if st.button("Сбросить к оригиналу", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.pop('export_cache', None)
            st.rerun()

        # ленивый экспорт без перегрузки памяти на холостых кликах
        st.markdown("---")
        st.markdown("### Экспорт данных")
        export_fmt = st.selectbox("Формат выгрузки", ["CSV", "Excel"], key="export_fmt_select")

        # генерируем файл только по нажатию кнопки подготовки
        if st.button("Сформировать файл для скачивания", use_container_width=True):
            with st.spinner("Сборка файла..."):
                file_bytes, mime_type, ext = export_dataframe_bytes(df_curr, export_fmt)
                st.session_state['export_cache'] = {
                    'data': file_bytes,
                    'mime': mime_type,
                    'filename': f"export_{st.session_state.filename or 'dataset'}{ext}"
                }

        if 'export_cache' in st.session_state:
            cached = st.session_state['export_cache']
            st.download_button(
                label=f"Скачать {cached['filename']}",
                data=cached['data'],
                file_name=cached['filename'],
                mime=cached['mime'],
                use_container_width=True
            )


# основной экран приложения
if st.session_state.df is None:
    # чистый лендинг без ии-клише
    st.markdown("""
    <div class="welcome-panel">
        <h2 style="margin: 0 0 8px 0; font-weight: 700; letter-spacing: -0.02em;">StatSnap</h2>
        <div style="font-size: 14px; opacity: 0.85;">
            Локальный аналитический комплекс экспресс-анализа, статистического моделирования и продуктовых метрик.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="hero-card">
            <span class="hero-tag">АНАЛИЗ И ГРАФИКИ</span>
            <div class="hero-title">Интерактивная визуализация</div>
            <div class="hero-desc">
                Многомерные проекции, автоматическое сэмплирование для защиты от зависания браузера, 
                матрица попарных корреляций, гистограммы распределений и боксплоты.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="hero-card">
            <span class="hero-tag">МОДЕЛИ И СТАТИСТИКА</span>
            <div class="hero-title">Машинное обучение</div>
            <div class="hero-desc">
                Регрессия OLS с p-value и проверкой значимости параметров, логистическая классификация, 
                деревья решений, ансамбли (Random Forest, Gradient Boosting) и K-Means.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="hero-card">
            <span class="hero-tag">ПРОДУКТОВЫЙ БЛОК</span>
            <div class="hero-title">Бизнес-аналитика</div>
            <div class="hero-desc">
                Когортный анализ с визуализацией оттока, N+1 Retention, 
                корректная RFM-сегментация без вылетов на дубликатах, воронка конверсии и расчет LTV.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("#### Быстрый старт")
    st.markdown("""
    1. Загрузите файл через левую панель в формате `.csv`, `.tsv`, `.xlsx` или `.xls`.
    2. Приложение автоматически определит разделитель, восстановит кодировку (включая кириллический `Windows-1251`) и проведет даункастинг типов данных для экономии RAM.
    3. Выберите соответствующий раздел для работы с выборкой.
    """)

else:
    df = st.session_state.df

    # навигационные вкладки
    tab_data, tab_viz, tab_ml, tab_metrics, tab_docs = st.tabs([
        "Таблица данных",
        "Визуализация",
        "Моделирование",
        "Бизнес-метрики",
        "Справочник pandas"
    ])

    # ---------------------------------------------------------
    # вкладка 1: таблица данных и инспектор типов
    # ---------------------------------------------------------
    with tab_data:
        st.markdown("#### Просмотр выборки")
        st.dataframe(df, use_container_width=True, height=360)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Структура признаков")
            col_info = []
            total_rows = len(df)
            for c in df.columns:
                nulls = int(df[c].isnull().sum())
                null_pct = (nulls / total_rows * 100) if total_rows > 0 else 0
                col_info.append({
                    "Колонка": c,
                    "Тип": str(df[c].dtype),
                    "Пропуски": f"{nulls} ({null_pct:.1f}%)",
                    "Уникальных": df[c].nunique(dropna=True)
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True, height=260)

        with col_right:
            st.markdown("#### Инструменты очистки")
            with st.expander("Удаление дубликатов", expanded=False):
                dups = int(df.duplicated().sum())
                st.write(f"Обнаружено полных дубликатов строк: **{dups}**")
                if st.button("Удалить дубликаты строк", disabled=(dups == 0)):
                    st.session_state.df = df.drop_duplicates().reset_index(drop=True)
                    st.success(f"Удалено строк: {dups}")
                    st.rerun()

            with st.expander("Обработка пропусков (NA)", expanded=False):
                na_col = st.selectbox("Колонка с пропусками", [c for c in df.columns if df[c].isnull().any()] or ["Нет колонок с NA"])
                if na_col != "Нет колонок с NA":
                    na_method = st.radio("Стратегия заполнения", ["Медиана", "Среднее", "Константа 0", "Удалить строки с NA"], horizontal=True)
                    if st.button("Применить к колонке"):
                        df_mod = df.copy()
                        if na_method == "Медиана" and pd.api.types.is_numeric_dtype(df_mod[na_col]):
                            df_mod[na_col] = df_mod[na_col].fillna(df_mod[na_col].median())
                        elif na_method == "Среднее" and pd.api.types.is_numeric_dtype(df_mod[na_col]):
                            df_mod[na_col] = df_mod[na_col].fillna(df_mod[na_col].mean())
                        elif na_method == "Константа 0":
                            df_mod[na_col] = df_mod[na_col].fillna(0)
                        elif na_method == "Удалить строки с NA":
                            df_mod = df_mod.dropna(subset=[na_col])
                        st.session_state.df = df_mod
                        st.success(f"Обработана колонка: {na_col}")
                        st.rerun()

            with st.expander("Удаление колонки", expanded=False):
                col_to_drop = st.selectbox("Выберите колонку для удаления", df.columns)
                if st.button("Удалить колонку"):
                    st.session_state.df = df.drop(columns=[col_to_drop])
                    st.success(f"Колонка {col_to_drop} удалена")
                    st.rerun()

    # ---------------------------------------------------------
    # вкладка 2: визуализация
    # ---------------------------------------------------------
    with tab_viz:
        st.markdown("#### Экспресс-визуализация")

        v_col1, v_col2 = st.columns([1, 2])

        with v_col1:
            chart_type = st.selectbox(
                "Тип графика",
                ["Диаграмма рассеяния (Scatter)", "Гистограмма", "Boxplot", "Линейный тренд", "Круговая диаграмма (Категории)"]
            )
            target_col = st.selectbox("Основная переменная (X / Значение)", df.columns, key="v_target")

            y_col = None
            if chart_type in ["Диаграмма рассеяния (Scatter)", "Линейный тренд"]:
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                y_col = st.selectbox("Вторая переменная (Y)", [c for c in num_cols if c != target_col] or df.columns.tolist(), key="v_y")

            group_choice = st.selectbox("Группировка по категории", ["Без группировки"] + list(df.columns), key="v_group")
            group_col = None if group_choice == "Без группировки" else group_choice

        with v_col2:
            # сэмплирование для plotly чтобы не забивать память браузера
            max_plot_points = 5000
            needs_sampling = len(df) > max_plot_points
            plot_df = df.sample(max_plot_points, random_state=42) if needs_sampling else df

            if needs_sampling and chart_type in ["Диаграмма рассеяния (Scatter)", "Boxplot"]:
                st.caption(f"Для сохранения плавности интерфейса отображено {max_plot_points:,} сэмплированных точек из {len(df):,}")

            fig = None
            if chart_type == "Диаграмма рассеяния (Scatter)" and y_col:
                fig = px.scatter(
                    plot_df, x=target_col, y=y_col, color=group_col,
                    template="plotly_white",
                    title=f"Взаимосвязь: {target_col} и {y_col}"
                )
                if pd.api.types.is_numeric_dtype(df[target_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    valid_data = df[[target_col, y_col]].dropna()
                    if len(valid_data) > 1:
                        corr_val = valid_data[target_col].corr(valid_data[y_col])
                        st.metric("Коэффициент корреляции Пирсона", f"{corr_val:.4f}")

            elif chart_type == "Гистограмма":
                fig = px.histogram(
                    plot_df, x=target_col, color=group_col,
                    barmode="overlay" if group_col else "relative",
                    template="plotly_white",
                    title=f"Распределение признака: {target_col}"
                )

            elif chart_type == "Boxplot":
                fig = px.box(
                    plot_df, y=target_col, x=group_col, color=group_col,
                    template="plotly_white",
                    title=f"Диаграмма размаха: {target_col}"
                )

            elif chart_type == "Линейный тренд" and y_col:
                fig = px.line(
                    plot_df.sort_values(by=target_col), x=target_col, y=y_col, color=group_col,
                    template="plotly_white",
                    title=f"Тренд: {y_col} относительно {target_col}"
                )

            elif chart_type == "Круговая диаграмма (Категории)":
                val_counts = df[target_col].value_counts().head(20).reset_index()
                val_counts.columns = [target_col, 'count']
                fig = px.pie(
                    val_counts, names=target_col, values='count',
                    template="plotly_white",
                    title=f"Топ-20 категорий: {target_col}"
                )

            if fig:
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Корреляционный анализ")
        num_features = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_features) >= 2:
            default_cols = num_features[:min(8, len(num_features))]
            selected_corr = st.multiselect("Признаки для матрицы", num_features, default=default_cols)
            if len(selected_corr) >= 2:
                corr_sub = df[selected_corr].corr()
                fig_corr = px.imshow(
                    corr_sub,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    template="plotly_white"
                )
                fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=380)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("В датасете недостаточно числовых полей для построения матрицы корреляции.")

    # ---------------------------------------------------------
    # вкладка 3: математические модели и машинное обучение
    # ---------------------------------------------------------
    with tab_ml:
        st.markdown("#### Обучение и валидация моделей")

        m_type = st.selectbox(
            "Алгоритм моделирования",
            [
                "Линейная регрессия (OLS с p-value)",
                "Логистическая регрессия",
                "Дерево решений (Decision Tree)",
                "Случайный лес (Random Forest)",
                "Градиентный бустинг (Gradient Boosting)",
                "Кластеризация (K-Means)"
            ]
        )

        all_numeric = df.select_dtypes(include=['number']).columns.tolist()

        if m_type != "Кластеризация (K-Means)":
            if len(df.columns) < 2:
                st.warning("Недостаточно колонок для построения предсказательной модели.")
            else:
                col_m1, col_m2 = st.columns([1, 2])

                with col_m1:
                    target_var = st.selectbox("Целевая переменная (Y)", df.columns, key="ml_target")
                    available_features = [c for c in df.columns if c != target_var]

                    # автоматическое предложение только числовых колонок для признаков
                    default_feats = [c for c in available_features if c in all_numeric][:5]
                    feature_vars = st.multiselect("Входные признаки (X)", available_features, default=default_feats, key="ml_feats")

                    use_split = st.checkbox("Валидационная выборка (Train / Test 80/20)", value=True)

                    # явный выбор типа задачи для деревьев и бустинга
                    task_kind = "Авто"
                    if m_type in ["Дерево решений (Decision Tree)", "Случайный лес (Random Forest)", "Градиентный бустинг (Gradient Boosting)"]:
                        task_kind = st.radio("Тип задачи", ["Авто", "Классификация", "Регрессия"], horizontal=True)

                    btn_train = st.button("Запустить обучение", use_container_width=True)

                with col_m2:
                    if btn_train:
                        if not feature_vars:
                            st.error("Выберите хотя бы один признак в X.")
                        else:
                            # безопасная чистка NaN в признаках и таргете одновременно
                            model_data = df[feature_vars + [target_var]].dropna()
                            if len(model_data) < 10:
                                st.error("Слишком мало наблюдений после очистки пропусков.")
                            else:
                                X_raw = model_data[feature_vars]
                                y_raw = model_data[target_var]

                                # кодирование категориальных признаков в X
                                X_mat = pd.get_dummies(X_raw, drop_first=True)

                                # определение типа задачи
                                is_classification = False
                                if m_type == "Логистическая регрессия":
                                    is_classification = True
                                elif task_kind == "Классификация":
                                    is_classification = True
                                elif task_kind == "Регрессия":
                                    is_classification = False
                                else:
                                    is_classification = y_raw.nunique() <= 5 or not pd.api.types.is_numeric_dtype(y_raw)

                                # подготовка таргета
                                y_mat = y_raw
                                label_encoder = None
                                if is_classification:
                                    label_encoder = LabelEncoder()
                                    y_mat = label_encoder.fit_transform(y_raw.astype(str))

                                # разбивка на обучающую и тестовую выборки
                                if use_split and len(model_data) > 30:
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_mat, y_mat, test_size=0.2, random_state=42
                                    )
                                else:
                                    X_train, X_test, y_train, y_test = X_mat, X_mat, y_mat, y_mat

                                # запуск соответствующей модели
                                if m_type == "Линейная регрессия (OLS с p-value)":
                                    if is_classification:
                                        st.warning("Целевая переменная похожа на категорию. Для классификации используйте Логистическую регрессию.")
                                    reg = LinearRegression().fit(X_train, y_train)
                                    y_pred = reg.predict(X_test)

                                    # статистика statsmodels
                                    X_sm = sm.add_constant(X_train)
                                    sm_res = sm.OLS(y_train, X_sm).fit()

                                    c_res1, c_res2, c_res3 = st.columns(3)
                                    c_res1.metric("R² (Test)", f"{r2_score(y_test, y_pred):.4f}")
                                    c_res2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                                    c_res3.metric("R² Adj (Train)", f"{sm_res.rsquared_adj:.4f}")

                                    st.code(explain_linear_regression(sm_res, y_test, y_pred, X_mat.columns, target_var))

                                    # график факт против прогноза
                                    fig_pred = px.scatter(
                                        x=y_test, y=y_pred,
                                        labels={'x': 'Фактические значения', 'y': 'Предсказанные значения'},
                                        title="Сравнение факта и прогноза",
                                        template="plotly_white"
                                    )
                                    fig_pred.add_trace(go.Scatter(
                                        x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                        mode='lines', line=dict(color='red', dash='dash'), name='Идеальный прогноз'
                                    ))
                                    fig_pred.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                                    st.plotly_chart(fig_pred, use_container_width=True)

                                elif m_type == "Логистическая регрессия":
                                    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                                    y_pred = clf.predict(X_test)

                                    acc = accuracy_score(y_test, y_pred)
                                    st.metric("Точность (Accuracy)", f"{acc:.4f}")

                                    # roc-auc для бинарной задачи
                                    if len(np.unique(y_mat)) == 2:
                                        y_proba = clf.predict_proba(X_test)[:, 1]
                                        auc = roc_auc_score(y_test, y_proba)
                                        st.metric("ROC-AUC", f"{auc:.4f}")

                                    st.text("Подробный классификационный отчет:")
                                    target_names = [str(c) for c in label_encoder.classes_] if label_encoder else None
                                    st.code(classification_report(y_test, y_pred, target_names=target_names))

                                else:
                                    # деревья, леса и бустинги
                                    model = None
                                    if m_type == "Дерево решений (Decision Tree)":
                                        model = DecisionTreeClassifier(max_depth=5) if is_classification else DecisionTreeRegressor(max_depth=5)
                                    elif m_type == "Случайный лес (Random Forest)":
                                        model = RandomForestClassifier(n_estimators=100, max_depth=8) if is_classification else RandomForestRegressor(n_estimators=100, max_depth=8)
                                    elif m_type == "Градиентный бустинг (Gradient Boosting)":
                                        model = GradientBoostingClassifier(n_estimators=100) if is_classification else GradientBoostingRegressor(n_estimators=100)

                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)

                                    if is_classification:
                                        st.metric("Accuracy (Test)", f"{accuracy_score(y_test, y_pred):.4f}")
                                        target_names = [str(c) for c in label_encoder.classes_] if label_encoder else None
                                        st.code(classification_report(y_test, y_pred, target_names=target_names))
                                    else:
                                        st.metric("R² (Test)", f"{r2_score(y_test, y_pred):.4f}")
                                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

                                    # график важности признаков
                                    importances = pd.DataFrame({
                                        'Признак': X_mat.columns,
                                        'Важность': model.feature_importances_
                                    }).sort_values(by='Важность', ascending=True)

                                    fig_imp = px.bar(
                                        importances, x='Важность', y='Признак', orientation='h',
                                        title="Важность признаков (Feature Importance)",
                                        template="plotly_white"
                                    )
                                    fig_imp.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                                    st.plotly_chart(fig_imp, use_container_width=True)

        else:
            # кластеризация k-means с сохранением результата
            st.markdown("##### Кластеризация K-Means")
            c_cols = st.multiselect("Признаки для кластеризации", all_numeric, default=all_numeric[:min(4, len(all_numeric))])
            n_clusters = st.slider("Число кластеров (k)", min_value=2, max_value=8, value=3)

            if len(c_cols) >= 2:
                if st.button("Рассчитать кластеры"):
                    clean_cl = df[c_cols].dropna()
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(clean_cl)
                    labels = km.labels_

                    st.session_state['last_cluster_result'] = {
                        'indices': clean_cl.index,
                        'labels': labels
                    }

                    # визуализация кластеров
                    df_viz = clean_cl.copy()
                    df_viz['Кластер'] = [f"Кластер {l}" for l in labels]

                    fig_km = px.scatter(
                        df_viz, x=c_cols[0], y=c_cols[1], color='Кластер',
                        title=f"Кластерная структура ({c_cols[0]} vs {c_cols[1]})",
                        template="plotly_white"
                    )
                    fig_km.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_km, use_container_width=True)

                # кнопка сохранения метки кластера в основной датасет
                if 'last_cluster_result' in st.session_state:
                    cl_info = st.session_state['last_cluster_result']
                    if st.button("Сохранить колонку 'cluster' в текущую таблицу"):
                        st.session_state.df['cluster'] = -1
                        st.session_state.df.loc[cl_info['indices'], 'cluster'] = cl_info['labels']
                        st.success("Метки кластеров успешно интегрированы в текущий датасет.")
                        st.session_state.pop('last_cluster_result', None)
                        st.rerun()
            else:
                st.info("Для кластеризации выберите не менее 2 числовых колонок.")

    # ---------------------------------------------------------
    # вкладка 4: бизнес-метрики
    # ---------------------------------------------------------
    with tab_metrics:
        st.markdown("#### Продуктовая и коммерческая аналитика")

        metric_kind = st.selectbox(
            "Тип анализа",
            [
                "N+1 Retention (Возвращаемость следующего дня)",
                "Когортный анализ (Cohort Analysis)",
                "RFM-сегментация",
                "Воронка конверсии",
                "LTV (Lifetime Value) и Churn Rate"
            ]
        )

        if metric_kind == "N+1 Retention (Возвращаемость следующего дня)":
            st.markdown("Оценка доли пользователей, совершивших повторное действие ровно на следующий день после первого.")
            col_u, col_d = st.columns(2)
            with col_u:
                user_id_col = st.selectbox("ID пользователя / клиента", df.columns, key="ret_u")
            with col_d:
                date_col = st.selectbox("Дата события", df.columns, key="ret_d")

            if st.button("Рассчитать Retention"):
                df_sub = df[[user_id_col, date_col]].dropna().copy()
                df_sub[date_col] = pd.to_datetime(df_sub[date_col], errors='coerce')
                df_sub = df_sub.dropna(subset=[date_col])

                first_dates = df_sub.groupby(user_id_col)[date_col].min().reset_index()
                first_dates.columns = [user_id_col, 'first_date']

                merged = df_sub.merge(first_dates, on=user_id_col)
                merged['day_diff'] = (merged[date_col].dt.floor('d') - merged['first_date'].dt.floor('d')).dt.days

                retention_users = merged[merged['day_diff'] == 1][user_id_col].nunique()
                total_base = first_dates[user_id_col].nunique()
                ret_rate = (retention_users / total_base * 100) if total_base > 0 else 0

                c_m1, c_m2 = st.columns(2)
                c_m1.metric("N+1 Retention", f"{ret_rate:.2f}%")
                c_m2.metric("Вернулись на день 1", f"{retention_users:,} из {total_base:,}")

        elif metric_kind == "Когортный анализ (Cohort Analysis)":
            col_cu, col_cd = st.columns(2)
            with col_cu:
                c_user_col = st.selectbox("ID пользователя", df.columns, key="coh_u")
            with col_cd:
                c_date_col = st.selectbox("Дата транзакции / визита", df.columns, key="coh_d")

            if st.button("Сформировать когорты"):
                df_c = df[[c_user_col, c_date_col]].dropna().copy()
                df_c[c_date_col] = pd.to_datetime(df_c[c_date_col], errors='coerce')
                df_c = df_c.dropna(subset=[c_date_col])

                first_events = df_c.groupby(c_user_col)[c_date_col].min().reset_index()
                first_events['cohort'] = first_events[c_date_col].dt.to_period('M')

                df_c = df_c.merge(first_events[[c_user_col, 'cohort']], on=c_user_col)
                df_c['activity_period'] = df_c[c_date_col].dt.to_period('M')

                # разница в месяцах
                df_c['period_index'] = (df_c['activity_period'] - df_c['cohort']).apply(lambda x: x.n)

                cohort_counts = df_c.groupby(['cohort', 'period_index'])[c_user_col].nunique().unstack(fill_value=0)
                cohort_sizes = cohort_counts.iloc[:, 0]
                cohort_pct = cohort_counts.divide(cohort_sizes, axis=0) * 100

                st.dataframe(cohort_pct.style.format("{:.1f}%"), use_container_width=True)

                fig_coh = px.imshow(
                    cohort_pct,
                    labels=dict(x="Месяцев с момента первой активности", y="Когорта", color="Retention %"),
                    text_auto=".1f",
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_coh.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_coh, use_container_width=True)

        elif metric_kind == "RFM-сегментация":
            st.markdown("Сегментация клиентской базы по давности (Recency), частоте (Frequency) и сумме (Monetary).")
            c_rfm1, c_rfm2, c_rfm3 = st.columns(3)
            with c_rfm1:
                rfm_u = st.selectbox("ID клиента", df.columns, key="rfm_u")
            with c_rfm2:
                rfm_d = st.selectbox("Дата заказа", df.columns, key="rfm_d")
            with c_rfm3:
                num_only = df.select_dtypes(include=['number']).columns.tolist()
                rfm_m = st.selectbox("Сумма заказа", num_only or df.columns, key="rfm_m")

            if st.button("Построить RFM матрицу"):
                df_rfm = df[[rfm_u, rfm_d, rfm_m]].dropna().copy()
                df_rfm[rfm_d] = pd.to_datetime(df_rfm[rfm_d], errors='coerce')
                df_rfm = df_rfm.dropna(subset=[rfm_d])
                max_date = df_rfm[rfm_d].max()

                rfm_table = df_rfm.groupby(rfm_u).agg({
                    rfm_d: lambda x: (max_date - x.max()).days,
                    rfm_u: 'count',
                    rfm_m: 'sum'
                }).rename(columns={rfm_d: 'Recency', rfm_u: 'Frequency', rfm_m: 'Monetary'})

                # безопасный qcut с защитой от дублирующихся квантилей через rank
                rfm_table['R_score'] = pd.qcut(rfm_table['Recency'].rank(method='first'), 4, labels=['4', '3', '2', '1'])
                rfm_table['F_score'] = pd.qcut(rfm_table['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
                rfm_table['M_score'] = pd.qcut(rfm_table['Monetary'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
                rfm_table['RFM_Segment'] = rfm_table['R_score'].astype(str) + rfm_table['F_score'].astype(str) + rfm_table['M_score'].astype(str)

                st.dataframe(rfm_table.head(100), use_container_width=True)

                seg_dist = rfm_table['RFM_Segment'].value_counts().head(15).reset_index()
                seg_dist.columns = ['Сегмент', 'Число клиентов']
                fig_rfm = px.bar(seg_dist, x='Сегмент', y='Число клиентов', template="plotly_white", title="Топ-15 RFM сегментов")
                fig_rfm.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_rfm, use_container_width=True)

        elif metric_kind == "Воронка конверсии":
            stage_col = st.selectbox("Колонка этапа воронки", df.columns, key="fnl_col")
            if st.button("Построить воронку"):
                counts = df[stage_col].value_counts().reset_index()
                counts.columns = ['Этап', 'Пользователей']

                fig_fnl = go.Figure(go.Funnel(
                    y=counts['Этап'],
                    x=counts['Пользователей'],
                    textinfo="value+percent initial"
                ))
                fig_fnl.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_fnl, use_container_width=True)

        elif metric_kind == "LTV (Lifetime Value) и Churn Rate":
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                ltv_u = st.selectbox("ID клиента", df.columns, key="ltv_u")
            with col_l2:
                num_only = df.select_dtypes(include=['number']).columns.tolist()
                ltv_m = st.selectbox("Выручка", num_only or df.columns, key="ltv_m")
            with col_l3:
                ltv_d = st.selectbox("Дата последней активности", df.columns, key="ltv_d")

            churn_days = st.slider("Период неактивности для признания оттока (дней)", 14, 180, 30)

            if st.button("Рассчитать LTV и Churn Rate"):
                df_sub = df[[ltv_u, ltv_m, ltv_d]].dropna().copy()
                df_sub[ltv_d] = pd.to_datetime(df_sub[ltv_d], errors='coerce')
                df_sub = df_sub.dropna(subset=[ltv_d])

                total_users = df_sub[ltv_u].nunique()
                total_revenue = df_sub[ltv_m].sum()
                avg_ltv = (total_revenue / total_users) if total_users > 0 else 0

                ref_date = df_sub[ltv_d].max()
                cutoff = ref_date - pd.Timedelta(days=churn_days)

                active_set = set(df_sub[df_sub[ltv_d] > cutoff][ltv_u].unique())
                all_set = set(df_sub[ltv_u].unique())
                churned_count = len(all_set - active_set)
                churn_rate = (churned_count / total_users * 100) if total_users > 0 else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Средний LTV на пользователя", f"{avg_ltv:,.2f}")
                c2.metric("Уровень оттока (Churn Rate)", f"{churn_rate:.1f}%")
                c3.metric("Ушедших пользователей", f"{churned_count:,} из {total_users:,}")

    # ---------------------------------------------------------
    # вкладка 5: справочник функций pandas
    # ---------------------------------------------------------
    with tab_docs:
        st.markdown("#### Интерактивный справочник по методам pandas")

        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            cat_list = ["Все категории"] + list(knowledge_base.keys())
            chosen_cat = st.selectbox("Категория операций", cat_list)
            search_query = st.text_input("Поиск по сигнатуре или описанию", placeholder="merge, dropna, qcut, filter...")

        with s_col2:
            query = search_query.strip().lower()
            active_cats = list(knowledge_base.keys()) if chosen_cat == "Все категории" else [chosen_cat]

            matched_items = []
            for cat in active_cats:
                for item in knowledge_base.get(cat, []):
                    if not query or (query in item['name'].lower() or query in item['desc'].lower() or query in item.get('example', '').lower()):
                        matched_items.append((cat, item))

            st.caption(f"Найдено методов: {len(matched_items)}")

            for cat_name, item in matched_items[:40]:
                with st.expander(f"`{item['name']}` — {item['desc']}"):
                    st.markdown(f"**Описание:** {item['desc']}")
                    st.code(item['example'], language='python')
                    if 'params' in item:
                        st.caption(f"Ключевые параметры: {item['params']}")
                    st.caption(f"Категория: {cat_name}")