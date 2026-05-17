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
    
    print(df.info())
    print(df.describe())

    # 2) Перевести datetime у тип datetime та зробити індексом.
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    # 3) Видалити повні дублі рядків.
    
    n_dups = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнити NaN у humidity_pct МЕДІАНОЮ ПО МІСЯЦЮ В МЕЖАХ МІСТА.
    #    Підказка: groupby([city, month]).transform('median'),
    #    де month = df.index.month.
    
    n_nan_before = df["humidity_pct"].isna().sum()
    df["month"] = df.index.month
    df["humidity_pct"] = df.groupby(["city", "month"])["humidity_pct"] \
        .transform(lambda s: s.fillna(s.median()))
    n_filled = n_nan_before - df["humidity_pct"].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Прибрати фізичні викиди:
    #    - temperature_c має бути в [-60, 60]
    #    - wind_speed_ms (де не NaN) має бути в [0, 60]
    mask = (
        (df["temperature_c"] >= -60) & (df["temperature_c"] <= 60) &
        (df["wind_speed_ms"].isna() | ((df["wind_speed_ms"] >= 0) & (df["wind_speed_ms"] <= 60)))
    )
    n_outliers = (~mask).sum()
    df = df[mask]
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
    
    by_city_temp = df.groupby("city")["temperature_c"].mean().sort_values(ascending=False)
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах. Хто найвологіше?
    
    by_city_precip = df.groupby("city")["precipitation_mm"].sum().sort_values(ascending=False)
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    # 3) Місячна середня температура: resample('ME').mean()
    #    (для старих pandas — 'M' замість 'ME').
    
    monthly_mean = df.resample('ME')['temperature_c'].mean()
    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # 4) Pivot: місто × місяць, значення = середня T.
    
    pivot = df.pivot_table(values="temperature_c", index="city", columns="month", aggfunc="mean")
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів з опадами > 5 мм по містах.
    #    Підказка: спочатку зробіть денні суми по місту, потім порахуйте.
    
    daily = df.groupby(["city", pd.Grouper(freq="D")])["precipitation_mm"].sum()
    rainy_days = (daily > 5).groupby(level="city").sum().astype(int)
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # 6) Знайти аномальний місяць.
    #    Підхід: для кожного календарного місяця (1..12) обчислити
    #    "норму" як середню по тому ж місяцю обох років, потім знайти
    #    (year, month) з максимальним |відхиленням| від норми.
    
    df["year"] = df.index.year
    monthly_avg = df.groupby(["year", "month"])["temperature_c"].mean()
    norm = monthly_avg.groupby(level="month").mean()
    deviation = monthly_avg - norm.reindex(monthly_avg.index, level="month")
    idx = deviation.abs().idxmax()
    anomaly_month = idx
    anomaly_dev = deviation[idx]
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
    
    fig, ax = plt.subplots(figsize=(11, 5))
    cities_3 = ["Київ", "Львів", "Одеса"]
    for city in cities_3:
        city_monthly = df[df["city"] == city]["temperature_c"].resample("ME").mean()
        ax.plot(city_monthly.index, city_monthly.values, marker="o", markersize=3, label=city)
    ax.set_title("Місячна динаміка температури по містах")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Температура (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 2: bar — сумарні опади по містах.
    
    fig, ax = plt.subplots(figsize=(8, 5))
    by_city_precip = analytics["by_city_precip"]
    ax.bar(by_city_precip.index, by_city_precip.values, color="steelblue")
    ax.set_title("Сумарні опади по містах")
    ax.set_xlabel("Місто")
    ax.set_ylabel("Опади (мм)")
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 3: hist — розподіл температур з вертикальними лініями
    #    mean і median.
    
    fig, ax = plt.subplots(figsize=(9, 5))
    temps = df["temperature_c"].dropna()
    ax.hist(temps, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(temps.mean(), color="red", linewidth=1.5, label=f"mean = {temps.mean():.1f}°C")
    ax.axvline(temps.median(), color="orange", linewidth=1.5, label=f"median = {temps.median():.1f}°C")
    ax.set_title("Розподіл температур")
    ax.set_xlabel("Температура (°C)")
    ax.set_ylabel("Кількість спостережень")
    ax.legend()
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 4: heatmap pivot місто × місяць (plt.imshow).
    #    Не забудьте colorbar і підписи осей.
    
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = analytics["pivot"]

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlBu_r"
    )

    fig.colorbar(im, ax=ax, label="Температура (°C)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Середня температура: місто × місяць")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Місто")
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
