import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fetch live historical data from the Open-Meteo API
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 19.11,
    "longitude": 72.92,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": ["wind_speed_10m", "surface_pressure", "temperature_2m", "precipitation"],
    "timezone": "auto"
}

print("Fetching historical weather records for Mumbai...")
response = requests.get(url, params=params)
data = response.json()

# 2. Process DataFrame
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
df.columns = ['Wind Speed (km/h)', 'Surface Pressure (hPa)', 'Temperature (°C)', 'Precipitation (mm)']

# Define meteorological seasons for the Indian Monsoon context
def get_monsoon_season(month):
    if month in [12, 1, 2]: return 'Winter (Dry/Cool)'
    elif month in [3, 4, 5]: return 'Pre-Monsoon (Hot/Dry)'
    elif month in [6, 7, 8, 9]: return 'Monsoon (SW Monsoon)'
    else: return 'Post-Monsoon'

df['Hour'] = df.index.hour
df['Month_Num'] = df.index.month
df['Month_Name'] = df.index.strftime('%b')
df['Season'] = df.index.month.map(get_monsoon_season)

# Create chronological daily averages for time-series tracking
df_daily = df.resample('D').mean(numeric_only=True)

# Plot 1: Pressure vs Wind Speed (Chronological Monsoon Engine)
fig, ax1 = plt.subplots(figsize=(14, 6))

# Wind Speed on Primary Y
color = '#1f77b4'
ax1.set_xlabel('Timeline (Chronological 2025)', fontsize=12)
ax1.set_ylabel('Daily Avg Wind Speed (km/h)', color=color, fontsize=12)
ax1.plot(df_daily.index, df_daily['Wind Speed (km/h)'], color=color, alpha=0.8, linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.3)

# Pressure on Secondary Y
ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Daily Avg Surface Pressure (hPa)', color=color, fontsize=12)
ax2.plot(df_daily.index, df_daily['Surface Pressure (hPa)'], color=color, alpha=0.7, linewidth=1.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Monsoon Transition: Surface Pressure Dip Driving Wind Speed Acceleration', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.show()

# Plot 2: Diurnal Wind Speed Volatility Profile by Season
fig, ax = plt.subplots(figsize=(12, 6))
season_order = ['Winter (Dry/Cool)', 'Pre-Monsoon (Hot/Dry)', 'Monsoon (SW Monsoon)', 'Post-Monsoon']

sns.lineplot(data=df, x='Hour', y='Wind Speed (km/h)', hue='Season', hue_order=season_order,
             palette='Set1', marker='o', linewidth=2, ax=ax)

ax.set_title('Diurnal Wind Speed Profiles: Macro-Scale Monsoon vs. Local Thermal Cycles', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Hour of Day (24hr)', fontsize=12)
ax.set_ylabel('Average Wind Speed (km/h)', fontsize=12)
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Meteorological Phase', loc='upper left')
fig.tight_layout()
plt.show()

# Plot 3: Precipitation vs Wind Speed Concurrency
fig, ax = plt.subplots(figsize=(11, 6))

# Filtering out absolute zero rain hours to focus purely on active weather events
df_rainy = df[df['Precipitation (mm)'] > 0.1]

scatter = ax.scatter(df_rainy['Precipitation (mm)'], df_rainy['Wind Speed (km/h)'],
                     c=df_rainy['Month_Num'], cmap='viridis', alpha=0.6, edgecolors='none', s=25)

ax.set_title('Impact of Wet Weather Events on Wind Speed Mechanics', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Hourly Precipitation Volume (mm)', fontsize=12)
ax.set_ylabel('Hourly Wind Speed (km/h)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)

# Create custom chronological colorbar matching calendar months
cbar = fig.colorbar(scatter, ax=ax, ticks=range(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
cbar.set_label('Time of Year (Chronological Progression)', fontsize=11)

fig.tight_layout()
plt.show()