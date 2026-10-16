# StatSnap

> Локальный веб-комплекс для экспресс-анализа, визуализации, статистического моделирования и продуктовых метрик.

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Release](https://img.shields.io/badge/release-Beta%20v0.1.0-059669?style=flat-square)](https://github.com/BearCodexx/StatSnap/releases)
[![License](https://img.shields.io/badge/license-MIT-2563EB?style=flat-square)](LICENSE)

</div>

---

## Ключевые возможности

| Функция | Описание |
|---|---|
| Оптимизация памяти (RAM) | Автоматический даункастинг числовых типов и перевод повторяющихся строк в категории со снижением нагрузки на память до 70-80% |
| Умная загрузка | Поточная детекция разделителей (запятая, точка с запятой, табуляция, пайп) и кодировок (UTF-8, UTF-8-SIG, CP1251) без дублирования файла в RAM |
| Интерактивная визуализация | Графики Plotly (диаграммы рассеяния, боксплоты, гистограммы, линейные тренды) с автоматическим сэмплированием больших выборок |
| Корреляционный анализ | Тепловая матрица Пирсона с вычислением коэффициентов и проверкой статистической значимости |
| Машинное обучение | Регрессия OLS с p-value и t-статистиками, логистическая регрессия, деревья решений, случайный лес и градиентный бустинг |
| Валидация моделей | Разделение Train/Test (80/20) с расчетом R², RMSE, MAE, Accuracy, ROC-AUC и полного классификационного отчета |
| Кластеризация K-Means | Многомерная кластеризация с прямой интеграцией меток кластеров в текущую рабочую таблицу |
| Продуктовые метрики | Расчет N+1 Retention, когортный анализ по месяцам, RFM-сегментация с защитой от коллизий квантилей, воронка конверсии, LTV и Churn Rate |
| Справочник pandas | Встроенная база знаний на 200+ методов с готовыми примерами кода и фильтрацией по 30 категориям |

---

## Архитектура проекта

```text
StatSnap/
├── .venv/                     # Виртуальное окружение Python
├── utils/                     # Вспомогательные модули обработки данных
│   ├── __init__.py            # Экспорт интерфейсов пакета
│   └── data_processing.py     # Парсинг кодировок, даункастинг типов и ленивый экспорт
├── app.py                     # Основное приложение и веб-интерфейс на Streamlit
├── knowledge_base.py          # База знаний и справочник методов pandas
├── requirements.txt           # Реестр зависимостей проекта
├── run.bat                    # Скрипт автоматического запуска для Windows
└── README.md                  # Документация проекта
```

---

## Пошаговое руководство

> [!NOTE]
> Архитектура приложения оптимизирована под работу с большими выборками при минимальном потреблении оперативной памяти.

### 1. Клонирование репозитория

```bash
git clone https://github.com/BearCodexx/StatSnap.git
cd StatSnap
```

### 2. Создание и активация виртуального окружения

На Windows:
```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
```

На Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск приложения

```bash
streamlit run app.py
```

> [!TIP]
> Для быстрого старта на Windows достаточно дважды кликнуть по файлу `run.bat` — он сам проверит окружение, установит недостающие пакеты и откроет браузер.

---

## Сводная таблица команд

| Действие | Команда | Назначение |
|---|---|---|
| Обычный запуск | `streamlit run app.py` | Запуск локального сервера по адресу `http://localhost:8501` |
| Запуск на кастомном порту | `streamlit run app.py --server.port 8080` | Переопределение рабочего сетевого порта |
| Автозапуск на Windows | `run.bat` | Инициализация venv и старт веб-сервера в один клик |
| Сброс данных | Кнопка «Сбросить к оригиналу» | Возврат датасета в исходное состояние без перезагрузки страницы |
| Выгрузка результатов | Кнопка «Сформировать файл» | Генерация чистого CSV или Excel на лету |

---

<div align="center">

Разработано: **BearCodex**  
Лицензия: **MIT**

</div>
