import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Tuple
import io


def prepare_regression_data(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[
    pd.DataFrame, pd.Series, List[str]]:
    """Подготавливает данные для регрессии"""
    df_clean = df.dropna(subset=[target_col])
    df_clean = df_clean.dropna(subset=feature_cols)

    X = df_clean[feature_cols]
    y = df_clean[target_col]
    X = sm.add_constant(X)

    return X, y, feature_cols


def run_ols_regression(X: pd.DataFrame, y: pd.Series):
    """Запускает OLS регрессию"""
    model = sm.OLS(y, X)
    results = model.fit()
    return results


def get_regression_report(results, feature_cols: List[str], target_col: str) -> str:
    """Возвращает текстовый отчёт о регрессии"""
    buffer = io.StringIO()

    buffer.write("=" * 50 + "\n")
    buffer.write(f"РЕГРЕССИОННЫЙ АНАЛИЗ\n")
    buffer.write(f"Целевая переменная: {target_col}\n")
    buffer.write("=" * 50 + "\n\n")

    buffer.write("📊 КАЧЕСТВО МОДЕЛИ:\n")
    buffer.write(f"   • R-squared (R²): {results.rsquared:.4f}\n")
    buffer.write(f"   • Adjusted R-squared: {results.rsquared_adj:.4f}\n")
    buffer.write(f"   • F-statistic: {results.fvalue:.2f}\n")
    buffer.write(f"   • Prob (F-statistic): {results.f_pvalue:.4e}\n\n")

    buffer.write("📈 КОЭФФИЦИЕНТЫ:\n")
    for var, coef, pval in zip(results.params.index, results.params.values, results.pvalues):
        stars = ""
        if pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"

        buffer.write(f"   • {var}: {coef:.4f} (p-value: {pval:.4e}) {stars}\n")
    buffer.write("\n")

    buffer.write("💡 ИНТЕРПРЕТАЦИЯ:\n")
    if results.rsquared > 0.7:
        buffer.write("   • Модель хорошо объясняет данные (R² > 0.7)\n")
    elif results.rsquared > 0.5:
        buffer.write("   • Модель средне объясняет данные (R² > 0.5)\n")
    else:
        buffer.write("   • Модель слабо объясняет данные (R² < 0.5)\n")

    sig_count = sum(1 for p in results.pvalues if p < 0.05)
    buffer.write(f"   • {sig_count} из {len(results.pvalues)} переменных статистически значимы (p < 0.05)\n")

    return buffer.getvalue()