import pandas as pd
import numpy as np
import io


def get_basic_info(df: pd.DataFrame) -> str:
    """Возвращает базовую информацию о датасете"""
    buffer = io.StringIO()

    buffer.write("=" * 50 + "\n")
    buffer.write("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ\n")
    buffer.write("=" * 50 + "\n\n")

    buffer.write(f"📊 Размерность: {df.shape[0]} строк × {df.shape[1]} колонок\n\n")

    # Типы данных
    buffer.write("📋 ТИПЫ ДАННЫХ:\n")
    for col, dtype in df.dtypes.items():
        buffer.write(f"   • {col}: {dtype}\n")
    buffer.write("\n")

    # Пропуски
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        buffer.write("⚠️ ПРОПУСКИ В ДАННЫХ:\n")
        for col, count in missing_cols.items():
            percent = (count / len(df)) * 100
            buffer.write(f"   • {col}: {count} ({percent:.1f}%)\n")
        buffer.write("\n")
    else:
        buffer.write("✅ Пропусков в данных нет\n\n")

    # Дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        buffer.write(f"⚠️ Дубликаты: {duplicates} строк\n\n")
    else:
        buffer.write("✅ Дубликатов нет\n\n")

    return buffer.getvalue()


def get_statistics(df: pd.DataFrame) -> str:
    """Возвращает статистики для числовых колонок"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return "⚠️ Нет числовых колонок для статистического анализа"

    buffer = io.StringIO()
    buffer.write("=" * 50 + "\n")
    buffer.write("СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ\n")
    buffer.write("=" * 50 + "\n\n")

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        buffer.write(f"📊 {col}:\n")
        buffer.write(f"   • Среднее (mean): {col_data.mean():.2f}\n")
        buffer.write(f"   • Медиана (median): {col_data.median():.2f}\n")
        buffer.write(f"   • Стандартное отклонение (std): {col_data.std():.2f}\n")
        buffer.write(f"   • Минимум (min): {col_data.min():.2f}\n")
        buffer.write(f"   • Максимум (max): {col_data.max():.2f}\n")
        buffer.write(f"   • 25% квартиль: {col_data.quantile(0.25):.2f}\n")
        buffer.write(f"   • 75% квартиль: {col_data.quantile(0.75):.2f}\n")
        buffer.write("\n")

    return buffer.getvalue()


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Возвращает список числовых колонок"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Возвращает список категориальных колонок"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_full_summary(df: pd.DataFrame) -> str:
    """Полная сводка"""
    return get_basic_info(df) + "\n" + get_statistics(df)