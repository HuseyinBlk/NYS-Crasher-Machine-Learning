import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set_theme(style="whitegrid", context="talk")
warnings.filterwarnings("ignore")


def save_fig(fig_path):
    try:
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    except:
        print("HATA!!")


df = pd.read_csv(config.RAW_DATA_FILE)
orig_columns = df.columns.tolist()
cols_map = {c: c.strip() for c in orig_columns}
df.rename(columns=cols_map, inplace=True)

missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage.sort_values(ascending=False)
missing_percentage = missing_percentage[missing_percentage > 0]

if not missing_percentage.empty:
    plt.figure(figsize=(12, 10))
    sns.barplot(
        y=missing_percentage.head(30).index,
        x=missing_percentage.head(30).values,
        palette='viridis'
    )
    plt.title('Ham Verideki En Çok Eksik Veriye Sahip İlk 30 Sütun')
    plt.xlabel('Eksik Veri Yüzdesi %')
    plt.ylabel('Sütun Adları')
    save_fig(config.FIGURES_DIR / 'raw_missing_data_top.png')

    cols_to_plot = missing_percentage.head(30).index
    df_missing_heatmap = df[cols_to_plot].isnull().astype(int)
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_missing_heatmap, cbar=False, cmap='rocket', yticklabels=False)
    plt.title('Eksik Veri Isı Haritası (En Çok Eksik 30 Sütun)')
    plt.xlabel('Sütunlar')
    plt.ylabel('Satırlar')
    save_fig(config.FIGURES_DIR / 'raw_missing_data_heatmap.png')

report_text = []
report_text.append(f"Toplam Satır: {df.shape[0]}")
report_text.append(f"Toplam Sütun: {df.shape[1]}")
report_text.append(f"Toplam Eksik Veri Sütunu: {len(missing_percentage)}")

numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 0:
    desc = df[numeric_cols].describe().T
    metrics_path = config.METRICS_DIR / "numeric_summary.csv"
    desc.to_csv(metrics_path)

if 'CRASH DATE' in df.columns:
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')
    df['year'] = df['CRASH DATE'].dt.year
    df['month'] = df['CRASH DATE'].dt.month
    df['weekday'] = df['CRASH DATE'].dt.day_name()

    yearly = df.groupby('year').size()
    if not yearly.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=yearly.index, y=yearly.values, palette='magma')
        plt.title('Yıllara Göre Kaza Sayısı')
        plt.xlabel('Yıl')
        plt.ylabel('Kaza Sayısı')
        save_fig(config.FIGURES_DIR / 'yearly_crash_distribution.png')

    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'weekday' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(x='weekday', data=df, order=order, palette='crest')
        plt.title('Haftanın Günlerine Göre Kaza Dağılımı')
        plt.xlabel('Gün')
        plt.ylabel('Kaza Sayısı')
        save_fig(config.FIGURES_DIR / 'weekday_crash_distribution.png')

if 'CRASH TIME' in df.columns:
    df['CRASH_TIME_parsed'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')
    df['CRASH_HOUR'] = df['CRASH_TIME_parsed'].dt.hour
    if df['CRASH_HOUR'].notnull().any():
        plt.figure(figsize=(12, 6))
        sns.countplot(x='CRASH_HOUR', data=df, palette='flare')
        plt.title('Saat Bazında Kaza Dağılımı')
        plt.xlabel('Saat (0–23)')
        plt.ylabel('Kaza Sayısı')
        save_fig(config.FIGURES_DIR / 'hourly_crash_distribution.png')

vehicle_col_candidates = [c for c in df.columns if 'VEHICLE' in c.upper() and 'TYPE' in c.upper()]
factor_col_candidates = [c for c in df.columns if 'CONTRIB' in c.upper() or 'CONTRIBUT' in c.upper()]

if vehicle_col_candidates:
    vehicle_col = vehicle_col_candidates[0]
    top_vehicles = df[vehicle_col].value_counts().dropna().head(10)
    if not top_vehicles.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(y=top_vehicles.index, x=top_vehicles.values, palette='rocket')
        plt.title('En Sık Kazaya Karışan 10 Araç Türü')
        plt.xlabel('Kaza Sayısı')
        plt.ylabel('Araç Türü')
        save_fig(config.FIGURES_DIR / 'top_vehicle_types.png')

if factor_col_candidates:
    factor_col = factor_col_candidates[0]
    top_factors = df[factor_col].value_counts().dropna().head(15)
    if not top_factors.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(y=top_factors.index, x=top_factors.values, palette='mako')
        plt.title('En Yaygın 15 Kaza Nedeni')
        plt.xlabel('Kaza Sayısı')
        plt.ylabel('Kaza Nedeni')
        save_fig(config.FIGURES_DIR / 'top_crash_factors.png')

inj_col = 'NUMBER OF PERSONS INJURED'
killed_col = 'NUMBER OF PERSONS KILLED'
cols_present = [c for c in [inj_col, killed_col] if c in df.columns]

if cols_present:
    totals = df[cols_present].sum()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=totals.index, y=totals.values, palette='coolwarm')
    plt.title('Toplam Yaralı ve Ölü Sayısı')
    plt.xlabel('Durum')
    plt.ylabel('Toplam Kişi Sayısı')
    save_fig(config.FIGURES_DIR / 'injury_death_total.png')

if 'BOROUGH' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y='BOROUGH', data=df, order=df['BOROUGH'].value_counts().index, palette='viridis')
    plt.title('Bölgelere Göre Kaza Sayısı')
    plt.xlabel('Kaza Sayısı')
    plt.ylabel('Borough')
    save_fig(config.FIGURES_DIR / 'borough_crash_distribution.png')

numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='crest', fmt=".2f")
    plt.title('Sayısal Değişkenler Arası Korelasyon')
    save_fig(config.FIGURES_DIR / 'numeric_correlation_heatmap.png')

summary_path = config.REPORTS_DIR / "summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Ham Veri Analizi Raporu ===\n\n")
    for line in report_text:
        f.write(line + "\n")
    f.write("\nÜretilen görseller:\n")
    files = list(config.FIGURES_DIR.glob('*'))
    for p in files:
        f.write(f"- {p.name}\n")
