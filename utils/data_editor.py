import pandas as pd
import re


def apply_edit(df: pd.DataFrame, instruction: str) -> tuple:
    """Применяет pandas код к DataFrame"""
    instruction = instruction.strip()
    result_msg = ""

    try:
        # Создаём безопасное окружение
        local_vars = {
            'df': df,
            'pd': pd,
            'np': __import__('numpy')
        }

        # Исполняем код
        exec(instruction, {'__builtins__': {}}, local_vars)

        # Получаем изменённый df
        new_df = local_vars.get('df', df)

        if new_df is None:
            return df, "❌ Переменная 'df' стала None"

        if len(new_df) == 0:
            return df, "❌ Датасет стал пустым"

        result_msg = f"✅ Изменения применены!\n📊 Было: {df.shape}, Стало: {new_df.shape}"
        return new_df, result_msg

    except Exception as e:
        return df, f"❌ Ошибка: {str(e)}"


def apply_simple_edit(df: pd.DataFrame, instruction: str) -> tuple:
    """Простые команды для тех, кто не знает pandas"""
    instruction = instruction.lower().strip()
    new_df = df.copy()
    result_msg = ""

    if instruction.startswith("drop rows where"):
        condition = instruction.replace("drop rows where", "").strip()
        try:
            before = len(new_df)
            new_df = new_df.query(condition)
            after = len(new_df)
            result_msg = f"✅ Удалено {before - after} строк"
        except Exception as e:
            result_msg = f"❌ Ошибка: {e}"

    elif instruction.startswith("fillna"):
        method = instruction.replace("fillna", "").strip()
        if method == "median":
            for col in new_df.select_dtypes(include=['number']).columns:
                new_df[col] = new_df[col].fillna(new_df[col].median())
            result_msg = "✅ Пропуски заполнены медианой"
        elif method == "mean":
            for col in new_df.select_dtypes(include=['number']).columns:
                new_df[col] = new_df[col].fillna(new_df[col].mean())
            result_msg = "✅ Пропуски заполнены средним"
        elif method == "0":
            new_df = new_df.fillna(0)
            result_msg = "✅ Пропуски заполнены нулями"
        else:
            result_msg = f"❌ Неизвестный метод: {method}"

    elif instruction.startswith("drop col"):
        col_name = instruction.replace("drop col", "").strip()
        if col_name in new_df.columns:
            new_df = new_df.drop(columns=[col_name])
            result_msg = f"✅ Удалена колонка: {col_name}"
        else:
            result_msg = f"❌ Колонка '{col_name}' не найдена"

    elif instruction.startswith("rename col"):
        match = re.search(r'rename col (.+) to (.+)', instruction)
        if match:
            old_name = match.group(1).strip()
            new_name = match.group(2).strip()
            if old_name in new_df.columns:
                new_df = new_df.rename(columns={old_name: new_name})
                result_msg = f"✅ Переименовано: {old_name} → {new_name}"
            else:
                result_msg = f"❌ Колонка '{old_name}' не найдена"
        else:
            result_msg = "❌ Формат: rename col старое to новое"

    elif instruction == "drop duplicates":
        before = len(new_df)
        new_df = new_df.drop_duplicates()
        after = len(new_df)
        result_msg = f"✅ Удалено {before - after} дубликатов"

    elif instruction == "reset index":
        new_df = new_df.reset_index(drop=True)
        result_msg = "✅ Индекс сброшен"

    else:
        # Пробуем как pandas код
        return apply_edit(df, instruction)

    return new_df, result_msg