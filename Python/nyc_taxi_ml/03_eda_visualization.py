"""
03_eda_visualization.py
--------------------------
Step 3 - Exploratory Data Analysis (EDA) with visualisations.
    
What this file does:
    - Generates 6 key plots that reveal data distribution and relationships
    - Uses Seaborn (static) + Ploty (interactive)
     - Save all figures to /data/plots/
        
Best practices
    - Every plot has a clear title, axis labels, and colour context
    - We examine: distribution, correlations, time patterns, spatial(location)
    - Insights are printed alongside each plot
"""
    
import os
import logging
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
    
from project_config import DATA_DIR, TARGET_COLUMN
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
    
# Seaborn them
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")

PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of trip duration (our target variable)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trip Duration Distibution", fontsize=15, fontweight="bold")
    
    # Raw distribution
    sns.histplot(df[TARGET_COLUMN], bins=60, kde=True, ax=axes[0], color=PALETTE[0])
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_title("Full Distribution")
    
    # Log-transformed
    sns.histplot(np.log1p(df[TARGET_COLUMN]), bins=60, kde=True, ax=axes[1], color=PALETTE[1])
    axes[1].set_xlabel("log(Duration + 1)")
    axes[1].set_title("Log-Transformed Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_target_distribution.png"), dpi=150)
    plt.close()
    log.info("Plot 1 saved: target disctribution")
    log.info(" Mean=%.1f min | Median=%.1f | Std=%.1f min",
             df[TARGET_COLUMN].mean(),
             df[TARGET_COLUMN].median(0),
             df[TARGET_COLUMN].std())
    
    

def plot_hourly_pattern(df: pd.DataFrame) -> None:
    """Average trip duration by hour of day - reveals rush-hour patterns."""
    hourly = df.groupby("pickup_hour")[TARGET_COLUMN].agg(["mean", "median", "std"]).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        hourly["pickup_hour"],
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.2, color=PALETTE[2], label="±1 std"
    )
    ax.plot(hourly["pickup_hour"], hourly["mean"], marker="o", color=PALETTE[2], label="Mean")
    ax.plot(hourly["pickup_hour"], hourly["median"], marker="s", color=PALETTE[3],
            linestyle="--", label="Median")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Trip duration (minutes)")
    ax.set_title("Average Trip Duration by Hour of Day", fontweight="bold")
    ax.set_xticks(range(24))
    ax.legend
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_hourly_patter.png"), dpi=150)
    plt.close()
    log.info("Plot 2 save: hourly patterns")
    
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation matrix - identify multicollinearuty"""
    num_cols = [
        TARGET_COLUMN, "trip_distance", "passenger_count",
        "fare_amount", "pickup_hour", "pickup_day_of_week"
    ]
    corr = df[num_cols].corr()
        
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # show lower trinagle only
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, square=True
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), dpi=150)
    plt.close()
    log.info("Plot 3 saved: correlation heatmap")
        
def plot_distance_vs_duration(df: pd.DataFrame) -> None:
    """Scatter: trip distance vs duration, coloured by hour."""
    sample = df.sample(min(10_000, len(df)), random_state=42)
    
    fig = px.scatter(
        sample,
        x="trip_distance",
        y=TARGET_COLUMN,
        color="pickup_hour",
        color_continuous_scale="Viridis",
        opacity=0.5,
        title="Trip Distance vs Duration (coloured by Hour)",
        labels={
            "trip_distance": "Distance (miles)",
            TARGET_COLUMN: "Duration (minutes)",
            "pickup_hour": "Hour",
        },
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=4))
    fig.write_html(os.path.join(PLOT_DIR, "04_distance_vs_duration.html"))
    log.info("Plot 4 save: interactive distance vs duration (HTML)")
    
def plot_weekly_heatmap(df: pd.DataFrame) -> None:
    """Hour * Day-of-week heatmap - reveals time patterns."""
    pivot = df.pivot_table(
        values=TARGET_COLUMN,
        index ="pickup_hour",
        columns="pickup_day_of_week",
        aggfunc="median"
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [day_names[d] for d in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        pivot, cmap="YlOrRd", linewidth=0.3,
        cbar_kws={"label": "Median Duration (min)"}, ax=ax
    )
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Median Trip Distance: Hour * Day of Week", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_weekly_heatmap.png"), dpi=150)
    plt.close()
    log.info("Plot 5 saved: weekly heatmap")
    
def plot_top_pickup_zones(df: DataFrame) -> None:
    """Bar chart: top 20 pickup zones by trip count"""
    top_zones = (
        df["pu_location_id"]
        .value_counts()
        .head(20)
        .reset_index()
    )
    top_zones.columns = ["pu_location_id", "count"]
    
    fig = px.bar(
        top_zones,
        x="count",
        y="pu_location_id",
        orientation="h",
        title="Top 20 Pickup Locations by Trip Count",
        labels={"pu_location_id": "Location ID", "count": "Trip Count"},
        color="count",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(os.path.join(PLOT_DIR, "06_top_pickup_zones.html"))
    log.info("Plot 6 saved: interactive top pickup zones (HTML)")
    

if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "clean_taxi_data.parquet")
    df = pd.read_parquet(clean_path)
    
    # Make sure time-based features exist for EDA
    if "pickup_hour" not in df.columns:
        df["pickup_hour"]        = df["pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
        
    log.info("Running EDA on %d rows ...", len(df))
    plot_target_distribution(df)
    plot_hourly_pattern(df)
    plot_correlation_heatmap(df)
    plot_distance_vs_duration(df)
    plot_weekly_heatmap(df)
    plot_top_pickup_zones(df)
    log.info("All plots saved to %d", PLOT_DIR)
    log.info("Step 3 complete ✓")