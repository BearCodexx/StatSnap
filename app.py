import streamlit as st  # веб морда
import pandas as pd  # таблички
import numpy as np  # математика
import matplotlib.pyplot as plt  # графики старые почти не юзаем
import seaborn as sns  # для красивых тепловых карт если что
import io  # байты в память
import os  # файлы хотя не особо нужно
import plotly.express as px  # интерактив визуализация
import plotly.graph_objects as go  # кастомные интерактивные графики
from plotly.subplots import make_subplots  # если надо несколько графиков в одном
from datetime import datetime, timedelta  # работа с датами
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler  # препроцессинг
from sklearn.linear_model import LinearRegression, LogisticRegression  # линейные модели
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, classification_report  # метрики качества
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor  # ансамбли
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # деревья
from sklearn.cluster import KMeans, DBSCAN  # кластеризация без учителя
import statsmodels.api as sm  # для p-value и нормальной статистики регрессии
import warnings  # затыкаем предупреждения

warnings.filterwarnings('ignore')

# настройка страницы стримлита широкая тема
st.set_page_config(
    page_title="StatSnap PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# кастомные стили карточек на главном экране
st.markdown("""
<style>
    .fixed-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .fixed-card-ml {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .fixed-card-metrics {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .fixed-card h2, .fixed-card-ml h2, .fixed-card-metrics h2 {
        color: white;
        margin-bottom: 10px;
    }
    .fixed-card p, .fixed-card-ml p, .fixed-card-metrics p {
        color: rgba(255,255,255,0.9);
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# инициализация состояния сессии храним датафрейм оригинал и имя файла
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'filename' not in st.session_state:
    st.session_state.filename = None


# функции

def detect_separator(uploaded_file):
    """определяем разделитель csv по первой строке"""
    content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    first_line = content.split('\n')[0]
    if ';' in first_line and ',' not in first_line:
        return ';'
    elif '\t' in first_line:
        return '\t'
    else:
        return ','


def load_data(uploaded_file):
    """загрузка csv или excel с автоопределением разделителя"""
    filename = uploaded_file.name
    if filename.endswith('.csv'):
        sep = detect_separator(uploaded_file)
        uploaded_file.seek(0)  # сброс указателя потому что выше читали
        return pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
    else:
        return pd.read_excel(uploaded_file)


def plot_interactive_chart(df, col, chart_type, color='blue', group_col=None):
    """строит интерактивный график plotly в зависимости от типа с группировкой или без"""
    if group_col and group_col != col:
        if chart_type == "Гистограмма":
            fig = px.histogram(df, x=col, color=group_col, title=f"Распределение: {col} по {group_col}",
                               barmode='group')
        elif chart_type == "Boxplot":
            fig = px.box(df, x=group_col, y=col, color=group_col, title=f"Boxplot: {col} по {group_col}")
        elif chart_type == "Круговая диаграмма":
            # для пайчарта с группировкой используем sunburst иерархия
            df_grouped = df.groupby([col, group_col]).size().reset_index(name='count')
            fig = px.sunburst(df_grouped, path=[col, group_col], values='count', title=f"{col} и {group_col}")
        else:
            fig = px.histogram(df, x=col, color=group_col, title=f"Распределение: {col} по {group_col}",
                               barmode='group')
    else:
        if chart_type == "Гистограмма":
            fig = px.histogram(df, x=col, title=f"Распределение: {col}", color_discrete_sequence=[color])
        elif chart_type == "Boxplot":
            fig = px.box(df, y=col, title=f"Boxplot: {col}", color_discrete_sequence=[color])
        elif chart_type == "Линейный график":
            if df[col].dtype in ['int64', 'float64']:
                fig = px.line(df, y=col, title=f"Тренд: {col}", color_discrete_sequence=[color])
            else:
                fig = px.bar(df[col].value_counts().head(30), title=f"Частоты: {col}", color_discrete_sequence=[color])
        elif chart_type == "Круговая диаграмма":
            val_counts = df[col].value_counts().head(30)  # режем до 30 чтоб не месиво
            fig = px.pie(values=val_counts.values, names=val_counts.index, title=f"Распределение: {col}")
        else:
            fig = px.histogram(df, x=col, title=f"Распределение: {col}", color_discrete_sequence=[color])
    return fig


def get_regression_report_full(results, X, y, feature_cols, target_col):
    """генерим человекочитаемый отчет по регрессии с интерпретацией p-value и R2"""
    report = ""
    report += "=" * 60 + "\n"
    report += f"РЕГРЕССИОННЫЙ АНАЛИЗ\n"
    report += f"Целевая переменная: {target_col}\n"
    report += "=" * 60 + "\n\n"

    # качество модели
    report += "📊 КАЧЕСТВО МОДЕЛИ:\n"
    report += f"   • R-squared (R²): {results.rsquared:.4f}\n"
    report += f"   • Adjusted R-squared: {results.rsquared_adj:.4f}\n"
    report += f"   • F-statistic: {results.fvalue:.2f}\n"
    report += f"   • Prob (F-statistic): {results.f_pvalue:.4e}\n"
    if results.f_pvalue < 0.05:
        report += "     ✅ Модель статистически значима (p < 0.05)\n"
    else:
        report += "     ❌ Модель статистически не значима (p > 0.05)\n"
    report += "\n"

    # коэффициенты с p-value и звездочками значимости
    report += "📈 КОЭФФИЦИЕНТЫ:\n"
    for var, coef, pval, std_err in zip(results.params.index, results.params.values, results.pvalues,
                                        results.bse.values):
        stars = ""
        if pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"

        sig_text = ""
        if pval < 0.05:
            sig_text = "✅ статистически значим"
        else:
            sig_text = "❌ не значим"

        report += f"   • {var}: {coef:.4f} (p-value: {pval:.4e}) {stars} — {sig_text}\n"
    report += "\n"

    # интерпретация R²
    report += "💡 ИНТЕРПРЕТАЦИЯ:\n"
    if results.rsquared > 0.9:
        report += "   • Модель отлично объясняет данные (R² > 0.9)\n"
    elif results.rsquared > 0.7:
        report += "   • Модель хорошо объясняет данные (R² > 0.7)\n"
    elif results.rsquared > 0.5:
        report += "   • Модель средне объясняет данные (R² > 0.5)\n"
    elif results.rsquared > 0.3:
        report += "   • Модель слабо объясняет данные (R² > 0.3)\n"
    else:
        report += "   • Модель очень слабо объясняет данные (R² < 0.3)\n"

    sig_count = sum(1 for p in results.pvalues if p < 0.05)
    total_vars = len(results.pvalues)
    report += f"   • {sig_count} из {total_vars} переменных статистически значимы (p < 0.05)\n"

    # уравнение регрессии
    report += "\n📐 УРАВНЕНИЕ РЕГРЕССИИ:\n"
    eq = f"{target_col} = {results.params['const']:.4f}"
    for var in feature_cols:
        if var in results.params.index:
            coef = results.params[var]
            sign = "+" if coef > 0 else "-"
            eq += f" {sign} {abs(coef):.4f}*{var}"
    report += f"   {eq}\n"

    return report


def replace_categorical_to_binary(df, col):
    """конвертим категорию в 0/1 если ровно 2 уникальных значения иначе предупреждение"""
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) == 2:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        return df[col].map(mapping), f"✅ Колонка '{col}' преобразована в бинарную (0/1): '{unique_vals[0]}'→0, '{unique_vals[1]}'→1"
    else:
        return df[col], f"⚠️ Колонка '{col}' имеет {len(unique_vals)} категорий. Бинаризация невозможна (нужно ровно 2)."


def replace_value_in_column(df, col, old_val, new_val):
    """простая замена значения в колонке"""
    return df[col].replace(old_val, new_val), f"✅ В колонке '{col}' заменено '{old_val}' → '{new_val}'"


# сайдбар
with st.sidebar:
    st.header("📁 Загрузка данных")

    uploaded_file = st.file_uploader(
        "CSV или Excel файл",
        type=['csv', 'xlsx', 'xls'],
        help="Без ограничений по размеру"
    )

    # планируемые обновления
    st.divider()
    st.subheader("🗓️ Планируемые обновления")

    if uploaded_file is None:  # показываем роадмап только если данных нет
        updates = {
            "Апрель 2026 (Pre-alpha v0.0.5)": [
                "✅ Интерактивные графики (plotly)",
                "✅ Математические модели (регрессия, классификация)",
                "✅ Бизнес-метрики (Retention, RFM, Cohort)",
                "✅ Размещение тестовой версии"
            ],
            "Май 2026 (Alpha v0.1.0)": [
                "⏳ Прогнозирование (ARIMA, Prophet)",
                "⏳ A/B тестирование",
                "⏳ Анализ временных рядов",
                "⏳ Исправление ошибок Pre-alpha"
            ],
            "Июнь 2026 (Alpha v0.1.x)": [
                "📅 Нейросетевые модели (MLP, LSTM)",
                "📅 Автоматический подбор гиперпараметров",
                "📅 Интерпретация моделей (SHAP, LIME)"
            ],
            "Июль 2026 (Alpha v0.1.x)": [
                "📅 Coming soon"
            ]
        }

        for month, items in updates.items():
            with st.expander(f"📌 {month}"):
                for item in items:
                    st.write(item)

        st.divider()

        # поддержка
        st.subheader("☕ Поддержать проект")
        st.markdown("""
            <div style="text-align: center;">
                <a href="https://www.donationalerts.com/r/bearcodexdev" target="_blank">
                    <button style="background: #FF9800; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%;">
                        💖 Поддержать разработчика
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Спасибо за поддержку! ❤️")

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.df_original = df.copy()  # сохраняем оригинал для сброса
            st.session_state.filename = uploaded_file.name
            st.success(f"✅ {df.shape[0]} строк × {df.shape[1]} колонок")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    if st.session_state.df is not None:
        st.divider()
        st.metric("📊 Строк", st.session_state.df.shape[0])
        st.metric("📋 Колонок", st.session_state.df.shape[1])

        if st.button("🔄 Сбросить к исходному"):
            st.session_state.df = st.session_state.df_original.copy()
            st.rerun()

    if st.session_state.df is not None:
        st.divider()

        # кнопка скачивания данных
        st.subheader("💾 Сохранить данные")

        file_format = st.selectbox("Формат", ["CSV", "Excel"], key="save_format")

        if file_format == "CSV":
            csv = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"{st.session_state.filename or 'data'}.csv",
                mime="text/csv",
                width='stretch'
            )
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Скачать Excel",
                data=excel_data,
                file_name=f"{st.session_state.filename or 'data'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )

        st.divider()

        # кнопка доната в сайдбаре
        st.subheader("☕ Поддержать проект")
        st.markdown("""
        <a href="https://www.donationalerts.com/r/bearcodexdev" target="_blank">
            <button style="background: #FF9800; color: white; border: none; padding: 10px; border-radius: 8px; width: 100%; cursor: pointer; font-size: 16px;">
                💖 Поддержать разработчика
            </button>
        </a>
        """, unsafe_allow_html=True)

# основной контент
if st.session_state.df is not None:
    df = st.session_state.df

    # вкладки основного интерфейса
    tabs = st.tabs([
        "📋 Данные",
        "📈 Визуализация",
        "🧠 Мат. модель",
        "📊 Метрики",
        "📚 Справочник pandas"
    ])

    # вкладка 1 (данные)
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Просмотр данных")
            st.dataframe(df, width='stretch')
        with col2:
            st.subheader("Типы данных")
            # Принудительно преобразуем всё в строки для совместимости
            dtype_data = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                dtype_data.append([col, col_type, missing_count, unique_count])

            dtype_df = pd.DataFrame(dtype_data, columns=['Колонка', 'Тип', 'Пропуски', 'Уникальные'])
            st.dataframe(dtype_df, width='stretch')

    # вкладка 2 (визуализация)
    with tabs[1]:
        st.subheader("Интерактивная визуализация")

        col1, col2 = st.columns([1, 2])

        with col1:
            chart_col = st.selectbox("Переменная", df.columns, key="chart_col")
            chart_type = st.selectbox(
                "Тип графика",
                ["Гистограмма", "Boxplot", "Линейный график", "Круговая диаграмма", "Scatter plot"],
                key="chart_type"
            )
            chart_color = st.color_picker("Цвет", "#1f77b4", key="chart_color")

            group_by = st.selectbox("Группировать по (опционально)", ["Нет"] + list(df.columns), key="group_by")
            group_col = None if group_by == "Нет" else group_by

        with col2:
            if chart_type == "Scatter plot":
                x_col = st.selectbox("Ось X", df.columns, key="scatter_x")
                y_col = st.selectbox("Ось Y", df.columns, key="scatter_y")
                if group_col:
                    fig = px.scatter(df, x=x_col, y=y_col, color=group_col, title=f"{y_col} vs {x_col} по {group_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                                     color_discrete_sequence=[chart_color])
                st.plotly_chart(fig, width='stretch')

                if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                    corr = df[[x_col, y_col]].corr().iloc[0, 1]
                    st.metric("Корреляция Пирсона", f"{corr:.3f}")

            else:
                fig = plot_interactive_chart(df, chart_col, chart_type, chart_color, group_col)
                if fig:
                    st.plotly_chart(fig, width='stretch')

        # матрица корреляций
        st.subheader("Матрица корреляций")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 1:
            selected_corr = st.multiselect("Выбери колонки для корреляции", numeric_cols,
                                           default=numeric_cols[:min(10, len(numeric_cols))], key="corr_cols")
            if len(selected_corr) >= 2:
                corr_matrix = df[selected_corr].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1,
                                zmax=1)
                fig.update_layout(title="Корреляционная матрица", height=500)
                st.plotly_chart(fig, width='stretch')

    # вкладка 3 (математические модели)
    with tabs[2]:
        st.subheader("Математические модели")

        model_type = st.selectbox(
            "Выбери модель",
            ["Линейная регрессия", "Логистическая регрессия", "Дерево решений", "Случайный лес", "Градиентный бустинг",
             "K-Means кластеризация"],
            key="model_type"
        )

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if model_type in ["Линейная регрессия", "Логистическая регрессия", "Дерево решений", "Случайный лес",
                          "Градиентный бустинг"]:
            if len(numeric_cols) >= 2:
                target = st.selectbox("Целевая переменная (Y)", numeric_cols, key="target")
                features = st.multiselect("Признаки (X)", [c for c in numeric_cols if c != target], key="features")

                if st.button("🚀 Обучить модель", key="train_model"):
                    with st.spinner("Обучаю..."):
                        X = df[features].dropna()
                        y = df.loc[X.index, target]  # синхронизируем индексы после дропа na

                        if model_type == "Линейная регрессия":
                            if y.dtype in ['int64', 'float64']:
                                model = LinearRegression()
                                model.fit(X, y)
                                y_pred = model.predict(X)

                                # для p-value юзаем statsmodels
                                X_sm = sm.add_constant(X)
                                sm_model = sm.OLS(y, X_sm).fit()

                                col1, col2, col3 = st.columns(3)
                                col1.metric("R²", f"{r2_score(y, y_pred):.4f}")
                                col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
                                col3.metric("Adj. R²", f"{sm_model.rsquared_adj:.4f}")

                                coef_data = []
                                for var, coef, pval in zip(sm_model.params.index, sm_model.params.values,
                                                           sm_model.pvalues):
                                    stars = ""
                                    if pval < 0.001:
                                        stars = "***"
                                    elif pval < 0.01:
                                        stars = "**"
                                    elif pval < 0.05:
                                        stars = "*"
                                    coef_data.append(
                                        {"Признак": var, "Коэффициент": coef, "p-value": pval, "Значимость": stars})
                                st.dataframe(pd.DataFrame(coef_data), width='stretch')

                                report = get_regression_report_full(sm_model, X_sm, y, features, target)
                                st.text(report)

                                fig = px.scatter(x=y, y=y_pred, title="Предсказанные vs Фактические")
                                fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                                              line=dict(color="red", dash="dash"))
                                st.plotly_chart(fig, width='stretch')
                            else:
                                st.error("Целевая переменная должна быть числовой для линейной регрессии")

                        elif model_type == "Логистическая регрессия":
                            if y.nunique() == 2:
                                model = LogisticRegression(max_iter=1000)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                y_proba = model.predict_proba(X)[:, 1]

                                col1, col2 = st.columns(2)
                                col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
                                col2.metric("ROC-AUC", f"{roc_auc_score(y, y_proba):.4f}")

                                coef_df = pd.DataFrame({'Признак': features, 'Коэффициент': model.coef_[0]})
                                st.dataframe(coef_df.sort_values('Коэффициент', key=abs, ascending=False),
                                             width='stretch')
                                st.text(classification_report(y, y_pred))
                            else:
                                st.error(f"Целевая переменная должна быть бинарной. Сейчас {y.nunique()} значений.")

                        elif model_type == "Дерево решений":
                            if y.dtype in ['int64', 'float64']:
                                model = DecisionTreeRegressor(max_depth=5)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("R²", f"{r2_score(y, y_pred):.4f}")
                            else:
                                model = DecisionTreeClassifier(max_depth=5)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")

                            importance_df = pd.DataFrame({'Признак': features, 'Важность': model.feature_importances_})
                            st.dataframe(importance_df.sort_values('Важность', ascending=False),
                                         width='stretch')

                            fig = px.bar(importance_df, x='Важность', y='Признак', orientation='h',
                                         title="Важность признаков")
                            st.plotly_chart(fig, width='stretch')

                        elif model_type == "Случайный лес":
                            if y.dtype in ['int64', 'float64']:
                                model = RandomForestRegressor(n_estimators=100, max_depth=10)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("R²", f"{r2_score(y, y_pred):.4f}")
                            else:
                                model = RandomForestClassifier(n_estimators=100, max_depth=10)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")

                            importance_df = pd.DataFrame({'Признак': features, 'Важность': model.feature_importances_})
                            st.dataframe(importance_df.sort_values('Важность', ascending=False),
                                         width='stretch')

                            fig = px.bar(importance_df, x='Важность', y='Признак', orientation='h',
                                         title="Важность признаков")
                            st.plotly_chart(fig, width='stretch')

                        elif model_type == "Градиентный бустинг":
                            if y.dtype in ['int64', 'float64']:
                                model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("R²", f"{r2_score(y, y_pred):.4f}")
                            else:
                                model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
                                model.fit(X, y)
                                y_pred = model.predict(X)
                                st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")

                            importance_df = pd.DataFrame({'Признак': features, 'Важность': model.feature_importances_})
                            st.dataframe(importance_df.sort_values('Важность', ascending=False),
                                         width='stretch')

                            fig = px.bar(importance_df, x='Важность', y='Признак', orientation='h',
                                         title="Важность признаков")
                            st.plotly_chart(fig, width='stretch')
            else:
                st.warning(f"Недостаточно числовых колонок. Найдено: {len(numeric_cols)}")

        elif model_type == "K-Means кластеризация":
            cluster_cols = st.multiselect("Колонки для кластеризации", numeric_cols, key="cluster_cols")
            n_clusters = st.slider("Количество кластеров", min_value=2, max_value=10, value=3, key="n_clusters")
            if st.button("Выполнить кластеризацию", key="cluster_btn") and len(cluster_cols) >= 2:
                X = df[cluster_cols].dropna()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_clustered = X.copy()
                df_clustered['Cluster'] = kmeans.fit_predict(X)
                st.dataframe(df_clustered['Cluster'].value_counts().reset_index(), width='stretch')
                fig = px.scatter(df_clustered, x=cluster_cols[0], y=cluster_cols[1], color='Cluster',
                                 title="Кластеризация")
                st.plotly_chart(fig, width='stretch')
                if st.button("Сохранить кластеры в датасет", key="save_clusters"):
                    st.session_state.df['Cluster'] = kmeans.predict(df[cluster_cols].fillna(0))
                    st.success("Колонка 'Cluster' добавлена в датасет")
                    st.rerun()

    # вкладка 4 (бизнес метрики)
    with tabs[3]:
        st.subheader("Бизнес-метрики")

        metric_type = st.selectbox(
            "Выбери метрику",
            ["Retention (N+1)", "Cohort анализ", "RFM анализ", "Конверсия в воронке", "LTV (Lifetime Value)",
             "Churn Rate"],
            key="metric_type"
        )

        if metric_type == "Retention (N+1)":
            st.markdown("""
            **Retention N+1** — процент пользователей, вернувшихся на следующий день.

            Требуемый формат:
            - Колонка с ID пользователя
            - Колонка с датой события
            """)

            user_col = st.selectbox("Колонка с ID пользователя", df.columns, key="ret_user")
            date_col = st.selectbox("Колонка с датой", df.columns, key="ret_date")

            if st.button("Рассчитать Retention", key="ret_btn"):
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    first_actions = df_temp.groupby(user_col)[date_col].min().reset_index()
                    first_actions.columns = [user_col, 'first_date']

                    df_temp['day_diff'] = (df_temp[date_col] - df_temp[user_col].map(
                        first_actions.set_index(user_col)['first_date'])).dt.days
                    retention_df = df_temp[df_temp['day_diff'] == 1].groupby(user_col).size().reset_index()
                    retention_rate = len(retention_df) / len(first_actions) * 100

                    st.metric("Retention N+1", f"{retention_rate:.2f}%")
                    st.success(f"Из {len(first_actions)} пользователей вернулись {len(retention_df)}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        elif metric_type == "Cohort анализ":
            st.info("Группировка пользователей по дате первого действия")
            user_col = st.selectbox("Колонка с ID пользователя", df.columns, key="cohort_user")
            date_col = st.selectbox("Колонка с датой", df.columns, key="cohort_date")

            if st.button("Построить Cohort", key="cohort_btn"):
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    first_actions = df_temp.groupby(user_col)[date_col].min().reset_index()
                    first_actions['cohort'] = first_actions[date_col].dt.strftime('%Y-%m')

                    df_temp = df_temp.merge(first_actions[[user_col, 'cohort']], on=user_col)
                    df_temp['period'] = (df_temp[date_col].dt.year - df_temp['cohort'].str[:4].astype(int)) * 12 + (
                                df_temp[date_col].dt.month - df_temp['cohort'].str[5:7].astype(int))

                    cohort_data = df_temp.groupby(['cohort', 'period'])[user_col].nunique().reset_index()
                    cohort_pivot = cohort_data.pivot(index='cohort', columns='period', values=user_col)

                    st.dataframe(cohort_pivot, width='stretch')

                    fig = px.imshow(cohort_pivot, text_auto=True, aspect="auto", color_continuous_scale="Greens")
                    fig.update_layout(title="Cohort анализ")
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        elif metric_type == "RFM анализ":
            st.markdown("**RFM** — Recency, Frequency, Monetary")
            user_col = st.selectbox("Колонка с ID пользователя", df.columns, key="rfm_user")
            date_col = st.selectbox("Колонка с датой", df.columns, key="rfm_date")
            value_col = st.selectbox("Колонка с суммой", df.select_dtypes(include=['number']).columns, key="rfm_value")

            if st.button("Рассчитать RFM", key="rfm_btn"):
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    now = df_temp[date_col].max()

                    rfm = df_temp.groupby(user_col).agg({
                        date_col: lambda x: (now - x.max()).days,
                        user_col: 'count',
                        value_col: 'sum'
                    }).rename(columns={date_col: 'Recency', user_col: 'Frequency', value_col: 'Monetary'})

                    st.dataframe(rfm, width='stretch')

                    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=['4', '3', '2', '1'])
                    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
                    rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
                    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(
                        str)

                    st.subheader("RFM сегменты")
                    st.dataframe(rfm['RFM_score'].value_counts(), width='stretch')
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        elif metric_type == "Конверсия в воронке":
            st.info("Анализ воронки: шаги → конверсия")
            step_col = st.selectbox("Колонка с шагами", df.columns, key="funnel_col")

            if st.button("Построить воронку", key="funnel_btn"):
                funnel = df[step_col].value_counts().reset_index()
                funnel.columns = ['Шаг', 'Количество']

                fig = go.Figure(go.Funnel(
                    y=funnel['Шаг'],
                    x=funnel['Количество'],
                    textposition="inside",
                    textinfo="value+percent initial"
                ))
                fig.update_layout(title="Воронка конверсии")
                st.plotly_chart(fig, width='stretch')

        elif metric_type == "LTV (Lifetime Value)":
            st.markdown("**LTV** — средняя выручка с одного пользователя за всё время")
            user_col = st.selectbox("Колонка с ID пользователя", df.columns, key="ltv_user")
            value_col = st.selectbox("Колонка с суммой", df.select_dtypes(include=['number']).columns, key="ltv_value")

            if st.button("Рассчитать LTV", key="ltv_btn"):
                ltv = df.groupby(user_col)[value_col].sum().mean()
                st.metric("Средний LTV", f"{ltv:.2f}")
                st.success("LTV = сумма всех покупок пользователя / количество пользователей")

        elif metric_type == "Churn Rate":
            st.markdown("**Churn Rate** — процент пользователей, которые прекратили использование")
            user_col = st.selectbox("Колонка с ID пользователя", df.columns, key="churn_user")
            date_col = st.selectbox("Колонка с датой", df.columns, key="churn_date")
            period = st.selectbox("Период для оттока", ["7 дней", "14 дней", "30 дней", "60 дней"], key="churn_period")

            if st.button("Рассчитать Churn Rate", key="churn_btn"):
                try:
                    days = int(period.split()[0])
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    now = df_temp[date_col].max()
                    cutoff = now - pd.Timedelta(days=days)

                    active_users = set(df_temp[df_temp[date_col] > cutoff][user_col].unique())
                    all_users = set(df_temp[user_col].unique())
                    churned = all_users - active_users
                    churn_rate = len(churned) / len(all_users) * 100

                    st.metric("Churn Rate", f"{churn_rate:.2f}%")
                    st.info(f"За {period}: {len(churned)} из {len(all_users)} пользователей не проявляли активность")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    # вкладка 5 (справочник пандас)
    with tabs[4]:
        st.subheader("📚 Справочник pandas")

        # Умный поиск
        search = st.text_input("🔍 Поиск (что хочешь сделать?)",
                               placeholder="например: фильтр, группировка, объединить таблицы, удалить пропуски...")

        # Категории
        category = st.selectbox(
            "📂 Категория",
            [
                "Все",
                "Чтение и запись",
                "Просмотр данных",
                "Фильтрация и сортировка",
                "Работа с колонками",
                "Очистка данных",
                "Агрегация и группировка",
                "Объединение данных",
                "Работа с датами",
                "Статистика"
            ]
        )

        # База знаний
        knowledge_base = {
            "Чтение и запись": [
                {"name": "pd.read_csv()", "desc": "Чтение CSV файла", "example": "df = pd.read_csv('file.csv')",
                 "params": "sep=';', encoding='utf-8', nrows=1000"},
                {"name": "pd.read_excel()", "desc": "Чтение Excel файла",
                 "example": "df = pd.read_excel('file.xlsx', sheet_name='Sheet1')"},
                {"name": "df.to_csv()", "desc": "Сохранить в CSV", "example": "df.to_csv('output.csv', index=False)"},
                {"name": "df.to_excel()", "desc": "Сохранить в Excel",
                 "example": "df.to_excel('output.xlsx', index=False)"},
            ],
            "Просмотр данных": [
                {"name": "df.head(n)", "desc": "Первые n строк", "example": "df.head(10)"},
                {"name": "df.tail(n)", "desc": "Последние n строк", "example": "df.tail(5)"},
                {"name": "df.sample(n)", "desc": "Случайные n строк", "example": "df.sample(100)"},
                {"name": "df.info()", "desc": "Информация о датасете (типы, память)", "example": "df.info()"},
                {"name": "df.describe()", "desc": "Статистика по числовым колонкам", "example": "df.describe()"},
                {"name": "df.shape", "desc": "Размерность (строки, колонки)", "example": "df.shape"},
                {"name": "df.columns", "desc": "Список колонок", "example": "df.columns.tolist()"},
                {"name": "df.dtypes", "desc": "Типы данных колонок", "example": "df.dtypes"},
                {"name": "df.nunique()", "desc": "Количество уникальных значений", "example": "df['city'].nunique()"},
                {"name": "df.value_counts()", "desc": "Частота значений", "example": "df['status'].value_counts()"},
            ],
            "Фильтрация и сортировка": [
                {"name": "df[условие]", "desc": "Фильтрация строк", "example": "df[df['age'] > 18]"},
                {"name": "df.query()", "desc": "Фильтрация через строку",
                 "example": "df.query('age > 18 and city == \"Moscow\"')"},
                {"name": "df.isin()", "desc": "Фильтр по списку значений",
                 "example": "df[df['city'].isin(['Moscow', 'SPb'])]"},
                {"name": "df.str.contains()", "desc": "Фильтр по тексту",
                 "example": "df[df['name'].str.contains('John')]"},
                {"name": "df.sort_values()", "desc": "Сортировка", "example": "df.sort_values('age', ascending=False)"},
                {"name": "df.nlargest()", "desc": "N самых больших", "example": "df.nlargest(10, 'salary')"},
                {"name": "df.nsmallest()", "desc": "N самых маленьких", "example": "df.nsmallest(5, 'price')"},
            ],
            "Работа с колонками": [
                {"name": "df['new'] = ...", "desc": "Создать новую колонку",
                 "example": "df['total'] = df['price'] * df['qty']"},
                {"name": "df.drop()", "desc": "Удалить колонку", "example": "df.drop('col', axis=1)"},
                {"name": "df.rename()", "desc": "Переименовать колонку",
                 "example": "df.rename(columns={'old': 'new'})"},
                {"name": "df.astype()", "desc": "Изменить тип колонки",
                 "example": "df['age'] = df['age'].astype('int')"},
                {"name": "pd.to_datetime()", "desc": "Преобразовать в дату",
                 "example": "df['date'] = pd.to_datetime(df['date'])"},
                {"name": "df.select_dtypes()", "desc": "Выбрать колонки по типу",
                 "example": "df.select_dtypes(include=['number'])"},
                {"name": "df[['col1', 'col2']]", "desc": "Выбрать несколько колонок", "example": "df[['name', 'age']]"},
            ],
            "Очистка данных": [
                {"name": "df.dropna()", "desc": "Удалить строки с пропусками", "example": "df.dropna()"},
                {"name": "df.fillna()", "desc": "Заполнить пропуски",
                 "example": "df.fillna(0) или df.fillna(df.mean())"},
                {"name": "df.interpolate()", "desc": "Интерполяция пропусков", "example": "df.interpolate()"},
                {"name": "df.drop_duplicates()", "desc": "Удалить дубликаты",
                 "example": "df.drop_duplicates(subset=['name'])"},
                {"name": "df.replace()", "desc": "Заменить значения", "example": "df['city'].replace('Msk', 'Moscow')"},
                {"name": "df.rename()", "desc": "Переименовать", "example": "df.rename(columns={'old': 'new'})"},
                {"name": "df.reset_index()", "desc": "Сбросить индекс", "example": "df.reset_index(drop=True)"},
                {"name": "df.apply()", "desc": "Применить функцию", "example": "df['age'].apply(lambda x: x*2)"},
            ],
            "Агрегация и группировка": [
                {"name": "df.groupby()", "desc": "Группировка", "example": "df.groupby('city')['sales'].sum()"},
                {"name": "df.pivot_table()", "desc": "Сводная таблица",
                 "example": "pd.pivot_table(df, values='sales', index='city', columns='product')"},
                {"name": "df.agg()", "desc": "Множественная агрегация",
                 "example": "df.groupby('city')['sales'].agg(['sum', 'mean', 'count'])"},
                {"name": "pd.crosstab()", "desc": "Таблица сопряжённости",
                 "example": "pd.crosstab(df['gender'], df['survived'])"},
                {"name": "df.rolling()", "desc": "Скользящее окно", "example": "df['sales'].rolling(7).mean()"},
                {"name": "df.cumsum()", "desc": "Кумулятивная сумма", "example": "df['sales'].cumsum()"},
                {"name": "df.rank()", "desc": "Ранжирование", "example": "df['score'].rank(ascending=False)"},
            ],
            "Объединение данных": [
                {"name": "pd.concat()", "desc": "Объединить по строкам", "example": "pd.concat([df1, df2])"},
                {"name": "pd.merge()", "desc": "Объединить по колонке (JOIN)",
                 "example": "pd.merge(df1, df2, on='id', how='left')"},
                {"name": "df.join()", "desc": "Объединить по индексу", "example": "df1.join(df2)"},
                {"name": "df.append()", "desc": "Добавить строки (устарело)", "example": "pd.concat([df1, df2])"},
            ],
            "Работа с датами": [
                {"name": "pd.to_datetime()", "desc": "Преобразовать в дату",
                 "example": "df['date'] = pd.to_datetime(df['date'])"},
                {"name": "df.dt.year", "desc": "Извлечь год", "example": "df['year'] = df['date'].dt.year"},
                {"name": "df.dt.month", "desc": "Извлечь месяц", "example": "df['month'] = df['date'].dt.month"},
                {"name": "df.dt.dayofweek", "desc": "День недели",
                 "example": "df['weekday'] = df['date'].dt.dayofweek"},
                {"name": "df.diff()", "desc": "Разница между датами",
                 "example": "df['days_diff'] = df['date'].diff().dt.days"},
                {"name": "pd.date_range()", "desc": "Создать диапазон дат",
                 "example": "pd.date_range('2024-01-01', periods=30, freq='D')"},
            ],
            "Статистика": [
                {"name": "df.mean()", "desc": "Среднее", "example": "df['age'].mean()"},
                {"name": "df.median()", "desc": "Медиана", "example": "df['salary'].median()"},
                {"name": "df.std()", "desc": "Стандартное отклонение", "example": "df['price'].std()"},
                {"name": "df.corr()", "desc": "Корреляция", "example": "df[['age', 'salary']].corr()"},
                {"name": "df.quantile()", "desc": "Квартиль", "example": "df['age'].quantile(0.75)"},
                {"name": "df.min() / df.max()", "desc": "Минимум / максимум",
                 "example": "df['price'].min(), df['price'].max()"},
            ],
        }


        # Функция поиска
        def search_functions(query, category):
            results = []
            query_lower = query.lower()

            cats = [category] if category != "Все" else list(knowledge_base.keys())

            for cat in cats:
                for func in knowledge_base.get(cat, []):
                    if (query_lower in func['name'].lower() or
                            query_lower in func['desc'].lower() or
                            query_lower in func['example'].lower()):
                        results.append((cat, func))
            return results


        # Отображение результатов
        if search:
            results = search_functions(search, category)

            if results:
                st.subheader(f"🔍 Найдено {len(results)} функций")
                for cat, func in results:
                    with st.expander(f"📌 {func['name']} — {func['desc']}"):
                        st.markdown(f"**Описание:** {func['desc']}")
                        st.markdown(f"**Пример:** `{func['example']}`")
                        if 'params' in func:
                            st.markdown(f"**Параметры:** {func['params']}")
                        st.markdown(f"**Категория:** {cat}")
                        if st.button(f"📋 Скопировать", key=f"copy_{func['name']}"):
                            st.code(func['example'], language='python')
            else:
                st.warning("Ничего не найдено. Попробуй другие ключевые слова.")

        else:
            # Показываем все функции по категориям
            cats = [category] if category != "Все" else list(knowledge_base.keys())

            for cat in cats:
                st.subheader(f"📁 {cat}")

                # Разбиваем на колонки для компактности
                cols = st.columns(2)
                for i, func in enumerate(knowledge_base[cat]):
                    with cols[i % 2]:
                        with st.expander(f"🔹 {func['name']}"):
                            st.markdown(f"**{func['desc']}**")
                            st.code(func['example'], language='python')
                            if 'params' in func:
                                st.caption(f"Параметры: {func['params']}")

else:
    # пустое состояние лендинг с карточками
    st.title("📊 StatSnap PRO")

    st.markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="color: white; margin-bottom: 10px;">😎 Профессиональный анализ данных</h1>
        <p style="color: white; font-size: 18px;">Загрузи данные и настраивай интерактивные графики, регрессию, бизнес-метрики</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; min-height: 220px; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="margin: 0 0 10px 0;">📈</h2>
            <h3 style="margin: 0 0 10px 0;">Визуализация</h3>
            <p style="margin: 0; font-size: 14px;">Гистограммы, boxplot'ы, scatter plot, круговые диаграммы, матрица корреляций</p>
            <small style="margin-top: 10px;">Интерактивные графики с zoom и hover</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; min-height: 220px; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="margin: 0 0 10px 0;">🧠</h2>
            <h3 style="margin: 0 0 10px 0;">Математические модели</h3>
            <p style="margin: 0; font-size: 14px;">Линейная регрессия, логистическая регрессия, дерево решений, случайный лес, бустинг, кластеризация</p>
            <small style="margin-top: 10px;">R², RMSE, коэффициенты, p-value</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; min-height: 220px; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="margin: 0 0 10px 0;">📊</h2>
            <h3 style="margin: 0 0 10px 0;">Бизнес-метрики</h3>
            <p style="margin: 0; font-size: 14px;">Retention N+1, Cohort анализ, RFM, воронка конверсии, LTV, Churn Rate</p>
            <small style="margin-top: 10px;">Профессиональные метрики для аналитики</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        ### 📁 Как начать

        1. Загрузить CSV или Excel файл в левой панели
        2. Выбрать нужную вкладку:
           - **Данные** — отслеживай данные датасета в реальном времени
           - **Визуализация** — построение интерактивных графиков
           - **Мат. модель** — обучение регрессии, классификации, кластеризации
           - **Метрики** — счет бизнес-показателей
        3. Наслаждаться комфортным, наглядным интерактивным анализом!
        """)

    with col_right:
        st.markdown("""
        ### 📊 Форматы данных

        Поддерживаются:
        - CSV (любые разделители: , ; \\t)
        - Excel (.xlsx, .xls)

        **Рекомендуется:** самостоятельно привести датасет в порядок
        """)

    st.divider()

    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>StatSnap PRO - локальный инструмент аналитика 🔥 | Created by BearCodex with Streamlit framework</p>
    </div>
    """, unsafe_allow_html=True)