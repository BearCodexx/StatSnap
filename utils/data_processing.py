import io
import csv
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def detect_separator_and_encoding(uploaded_file) -> Tuple[str, str]:
    # читаем только первый чанк для экономии ram вместо выгрузки всего файла
    raw_chunk = uploaded_file.read(8192)
    uploaded_file.seek(0)

    # определяем кодировку: сначала utf-8, fallback на cp1251 для кириллицы из excel
    encoding = 'utf-8'
    decoded_text = ""
    for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
        try:
            decoded_text = raw_chunk.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if not decoded_text:
        decoded_text = raw_chunk.decode('utf-8', errors='ignore')
        encoding = 'utf-8'

    first_line = decoded_text.splitlines()[0] if decoded_text.splitlines() else ""

    # подсчет вероятных разделителей
    candidates = [';', ',', '\t', '|']
    counts = {sep: first_line.count(sep) for sep in candidates}
    best_sep = max(counts, key=counts.get) if any(counts.values()) else ','

    return best_sep, encoding


def optimize_dataframe_memory(df: pd.DataFrame, convert_categories: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # замер исходной памяти
    initial_bytes = df.memory_usage(deep=True).sum()
    n_rows = len(df)

    if n_rows == 0:
        return df, {"initial_mb": 0.0, "final_mb": 0.0, "saved_pct": 0.0}

    # оптимизация числовых колонок
    for col in df.select_dtypes(include=['int', 'integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # оптимизация строковых колонок с низкой кардинальностью
    if convert_categories:
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            if num_unique / n_rows < 0.5 and num_unique < 500:
                try:
                    df[col] = df[col].astype('category')
                except Exception:
                    pass

    final_bytes = df.memory_usage(deep=True).sum()
    saved_bytes = max(0, initial_bytes - final_bytes)
    saved_pct = (saved_bytes / initial_bytes * 100) if initial_bytes > 0 else 0.0

    stats = {
        "initial_mb": initial_bytes / (1024 * 1024),
        "final_mb": final_bytes / (1024 * 1024),
        "saved_pct": saved_pct
    }
    return df, stats


def load_dataset(uploaded_file, optimize: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filename = uploaded_file.name.lower()

    if filename.endswith(('.csv', '.txt')):
        sep, encoding = detect_separator_and_encoding(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
    elif filename.endswith(('.xlsx', '.xls')):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Неподдерживаемый формат: {uploaded_file.name}")

    stats = {"initial_mb": df.memory_usage(deep=True).sum() / (1024 * 1024), "final_mb": 0.0, "saved_pct": 0.0}
    if optimize:
        df, stats = optimize_dataframe_memory(df)

    return df, stats


def export_dataframe_bytes(df: pd.DataFrame, file_format: str) -> Tuple[bytes, str, str]:
    # генерация байтов на лету без хранения промежуточных копий в state
    if file_format.upper() == "CSV":
        data = df.to_csv(index=False).encode('utf-8-sig')
        return data, "text/csv", ".csv"
    else:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
