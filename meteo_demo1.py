import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fetch data from Open-Meteo API
url = "https://archive-api.open-meteo.com/v1/archive"   # API endpoint
params = {
    "latitude": 19.11,
    "longitude": 72.92,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": ["wind_speed_10m", "surface_pressure", "temperature_2m", "precipitation"],
    "timezone": "auto"
}

print("Fetching data from API...")
response = requests.get(url, params=params)
data = response.json()

# 2. Data Preparation
df = pd.DataFrame(data["hourly"])       # Extract hourly data
df["time"] = pd.to_datetime(df["time"]) # Convert time to datetime
df.set_index("time", inplace=True)      # Set time as index

# Resample to Daily frequencies for smoother trend lines
df_daily = df.resample('D').agg({
    'temperature_2m': ['mean', 'min', 'max'],
    'wind_speed_10m': 'mean',
    'surface_pressure': 'mean'
})
# Flatten multi-index columns resulting from aggregation
df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]

# Resample precipitation to monthly totals
df_monthly = df.resample('M').agg({'precipitation': 'sum'})
df_monthly.index = df_monthly.index.strftime('%b') # Format index as short month names (Jan, Feb, etc.)

print("Generating visualizations...")

# ----------------- Plot 1: Daily Temperature Trends -----------------
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['temperature_2m_max'], label='Max Temperature', color='#d95f02', alpha=0.8)
plt.plot(df_daily.index, df_daily['temperature_2m_mean'], label='Mean Temperature', color='#7570b3', linewidth=2)
plt.plot(df_daily.index, df_daily['temperature_2m_min'], label='Min Temperature', color='#1f77b4', alpha=0.8)

plt.title('Mumbai 2025: Daily Temperature Trends', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Images/mumbai_temperature_2025.png', dpi=300)
plt.show()

# ----------------- Plot 2: Hourly/Monthly Temperature Heatmap -----------------
# Pivot raw hours vs months to get a 2D grid matrix of mean temperatures
pivot_df = df_meta.groupby(['Month', 'Hour'])['Temperature (°C)'].mean().unstack(level='Hour').reindex(month_chronological)

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(pivot_df, cmap='coolwarm', annot=False, cbar_kws={'label': 'Mean Temperature (°C)'}, ax=ax)
ax.set_title('Mumbai 2025: Hourly Temperature Heatmap Across Months', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Hour of the Day', fontsize=12)
ax.set_ylabel('Month', fontsize=12)
fig.tight_layout()
fig.savefig('Images/mumbai_hourly_monthly_heatmap.png', dpi=300)
plt.show()

# ----------------- Plot 3: Monthly Precipitation -----------------
plt.figure(figsize=(10, 5))
# Ensure the bars are in chronological order by month
sns.barplot(x=df_monthly.index, y=df_monthly['precipitation'], palette='Blues_r', hue=df_monthly['precipitation'], legend=False)

plt.title('Mumbai 2025: Monthly Total Precipitation', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Precipitation (mm)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Images/mumbai_precipitation_2025.png', dpi=300)
plt.show()

# ----------------- Plot 4: Wind Speed & Surface Pressure -----------------
fig, ax1 = plt.subplots(figsize=(12, 6))
# Primary Y-axis for Wind Speed
color_wind = '#2ca02c'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Daily Avg Wind Speed (km/h)', color=color_wind, fontsize=12)
ax1.plot(df_daily.index, df_daily['wind_speed_10m_mean'], color=color_wind, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color_wind)
ax1.grid(True, linestyle='--', alpha=0.3)

# Secondary Y-axis for Surface Pressure
ax2 = ax1.twinx()
color_press = '#9467bd'
ax2.set_ylabel('Daily Avg Surface Pressure (hPa)', color=color_press, fontsize=12)
ax2.plot(df_daily.index, df_daily['surface_pressure_mean'], color=color_press, alpha=0.6)
ax2.tick_params(axis='y', labelcolor=color_press)

plt.title('Mumbai 2025: Wind Speed & Surface Pressure Trends', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('Images/mumbai_wind_pressure_2025.png', dpi=300)
plt.show()

# ----------------- Plot 5: Correlation Matrix Heatmap -----------------
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df.corr()
# Mask upper triangle if you prefer a clean diagonal layout
sns.heatmap(corr_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title('Mumbai 2025: Correlation Matrix of Weather Variables', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
fig.savefig('Images/mumbai_weather_correlation.png', dpi=300)
plt.show()

print("All plots successfully generated and saved as PNG files!")