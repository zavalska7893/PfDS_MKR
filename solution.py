# ====================================================================
# Прізвище, ім'я, по батькові: Завальська Анастасія Вадимівна
# Група:                       КІ-33
# Дата виконання:              18.05.2026
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    """Підключитися до MySQL і завантажити таблицю observations.

    MySQL-контейнер на старті виконує LOAD DATA INFILE, що займає
    ~20–30 секунд. Тому робимо retry-цикл — перші спроби очікувано
    падають з OperationalError (server not ready).
    """
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


# ====================================================================
# БЛОК 1. NumPy (15 балів)
# ====================================================================
# Працюємо з СИРИМИ даними (до очищення в Pandas). Використовуємо
# тільки numpy-арифметику, без pandas-арифметики.

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    # 1) Побудувати np.array apparent temperature за формулою:
    #    T_app = T - (100 - RH) / 5
    #    Працюйте з temperature_c і humidity_pct як з np.array.
    
    T = df_raw["temperature_c"].to_numpy(dtype=float)
    RH = df_raw["humidity_pct"].to_numpy(dtype=float)
    apparent = T - (100 - RH) / 5
    print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

    # 2) Замінити викидні значення:
    #    - temperature_c > 60 або < -60   -> np.nan
    #    - wind_speed_ms > 100            -> np.nan
    #    Використати np.where.
    
    temperature_clean = np.where((df_raw["temperature_c"] > 60) | (df_raw["temperature_c"] < -60), np.nan, df_raw["temperature_c"])
    wind_clean = np.where(df_raw["wind_speed_ms"] > 100, np.nan, df_raw["wind_speed_ms"])
    print(f"2) Викидів температури замінено: {np.sum(np.isnan(temperature_clean))}")
    print(f"   Викидів вітру замінено:       {np.sum(np.isnan(wind_clean))}")

    # 3) Порахувати mean / median / std температури ВРУЧНУ
    #    (без pandas .describe(), ігноруючи NaN). Дозволені np.nansum,
    #    np.nanmedian, np.sqrt, маски тощо.
    
    mean_t = np.nanmean(temperature_clean)
    median_t = np.nanmedian(temperature_clean)
    std_t = np.nanstd(temperature_clean)
    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # 4) Маска: скільки спостережень "морозних" (T<0) і "жарких" (T>30).

    n_frost = np.sum(temperature_clean < 0)
    n_hot = np.sum(temperature_clean > 30)
    print(f"4) морозних: {n_frost}    жарких: {n_hot}")

    # 5) argmax / argmin температури -> повернути obs_id і datetime
    #    цих рядків. Підказка: np.nanargmax / np.nanargmin.
    
    idx_max = np.nanargmax(temperature_clean)
    idx_min = np.nanargmin(temperature_clean)
    print(f"5) Макс T: obs_id={df_raw['obs_id'].iloc[idx_max]}, datetime={df_raw['datetime'].iloc[idx_max]}, T={temperature_clean[idx_max]:.1f}°C")
    print(f"   Мін T:  obs_id={df_raw['obs_id'].iloc[idx_min]}, datetime={df_raw['datetime'].iloc[idx_min]}, T={temperature_clean[idx_min]:.1f}°C")


# ====================================================================
# БЛОК 2. Pandas — очищення (20 балів)
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # 1) Перевірте типи (info), статистику (describe).
    # TODO:
    pass

    # 2) Перевести datetime у тип datetime та зробити індексом.
    # TODO:
    pass

    # 3) Видалити повні дублі рядків.
    # TODO:
    n_dups = ...
    print(f"2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнити NaN у humidity_pct МЕДІАНОЮ ПО МІСЯЦЮ В МЕЖАХ МІСТА.
    #    Підказка: groupby([city, month]).transform('median'),
    #    де month = df.index.month.
    # TODO:
    n_filled = ...
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Прибрати фізичні викиди:
    #    - temperature_c має бути в [-60, 60]
    #    - wind_speed_ms (де не NaN) має бути в [0, 60]
    # TODO:
    n_outliers = ...
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    # 6) Звіт очищення.
    print(f"\n   Звіт: {rows_before} → {len(df)} рядків")

    return df


# ====================================================================
# БЛОК 3. Pandas — аналітика (30 балів)
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # 1) Середня температура по містах (sort_values).
    #    Хто найтепліше / найхолодніше?
    # TODO:
    by_city_temp = ...
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах. Хто найвологіше?
    # TODO:
    by_city_precip = ...
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    # 3) Місячна середня температура: resample('ME').mean()
    #    (для старих pandas — 'M' замість 'ME').
    # TODO:
    monthly_mean = ...
    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # 4) Pivot: місто × місяць, значення = середня T.
    # TODO:
    pivot = ...
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів з опадами > 5 мм по містах.
    #    Підказка: спочатку зробіть денні суми по місту, потім порахуйте.
    # TODO:
    rainy_days = ...
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # 6) Знайти аномальний місяць.
    #    Підхід: для кожного календарного місяця (1..12) обчислити
    #    "норму" як середню по тому ж місяцю обох років, потім знайти
    #    (year, month) з максимальним |відхиленням| від норми.
    # TODO:
    anomaly_month = ...
    anomaly_dev = ...
    print(f"\n6) Аномальний місяць: {anomaly_month}  відхилення = {anomaly_dev:+.2f}°C")

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
    }


# ====================================================================
# БЛОК 4. Matplotlib + інтерпретація (35 балів)
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # Графік 1: line — місячна динаміка температури по 3 обраних містах.
    # Вимоги: title, xlabel, ylabel, legend, форматування дат.
    # TODO:
    fig, ax = plt.subplots(figsize=(11, 5))
    # ... побудувати графік ...
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 2: bar — сумарні опади по містах.
    # TODO:
    fig, ax = plt.subplots(figsize=(8, 5))
    # ... побудувати графік ...
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 3: hist — розподіл температур з вертикальними лініями
    #    mean і median.
    # TODO:
    fig, ax = plt.subplots(figsize=(9, 5))
    # ... побудувати графік ...
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 4: heatmap pivot місто × місяць (plt.imshow).
    #    Не забудьте colorbar і підписи осей.
    # TODO:
    fig, ax = plt.subplots(figsize=(11, 5))
    # ... побудувати графік ...
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"4 графіки збережені в {PLOTS_DIR}/")


# ====================================================================

def main() -> None:
    df_raw = load_observations()
    print(f"Завантажено: shape={df_raw.shape}")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()


"""
ВИСНОВКИ (5–8 речень).

Напишіть тут вашу інтерпретацію даних. Орієнтири:
- Яке місто найтепліше/найхолодніше? Як ви це поясните географічно?
- Як виражена сезонність температури?
- Який місяць аномальний? Це хвиля спеки чи холоду? Як ви це визначили?
- Який кліматичний регіон стабільніший за температурою (за std)?
- 1–2 рекомендації: що б ви порадили на основі цих даних
  (наприклад, де варто будувати склади-холодильники, яку статтю
  витрат компанії важливо враховувати взимку тощо).

Ваш текст:
... ваш текст тут ...
"""
