import pandas as pd # NOQA F401
import numpy as np # NOQA F401
import matplotlib.pyplot as plt # NOQA F401
from matplotlib.animation import FuncAnimation # NOQA F401
from pathlib import Path # NOQA F401
import plotly.express as px # NOQA F401
# 2025 values are projected
df = pd.read_csv('/Users/conallnoonan/Documents/GDP_Analysis/2020-2025.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.Country.nunique())
# Which Country had the highest GDP in 2020?
print(df.sort_values(by='2020', ascending=False).head(1))
# What about 2025?
print(df.sort_values(by='2025', ascending=False).head(1))
# Which country had the lowest GDP in 2020?
print(df.sort_values(by='2020', ascending=True).head(1))
# What about 2025?
print(df.sort_values(by='2025', ascending=True).head(1))
print(df.head())
# Which country had the largest percent gain between 2020 and 2025?
df["2020"] = df["2020"].astype(float)
df["2025"] = df["2025"].astype(float)
df["2020-2025%"] = ((df["2025"] - df["2020"]) / df["2020"]) * 100
df["2020-2025%"] = round(df["2020-2025%"], 2)
print(df.sort_values(by="2020-2025%", ascending=False).head(1))
# Which countries had the largest drop in percentage of GDP between 2020 and
# 2025?
print(df.sort_values(by='2020-2025%', ascending=True).head(10))


# VISUALISATIONS


# Extract year columns and and convert country names to strings
df.columns = [str(c) for c in df.columns]
year_cols = [c for c in df.columns if c != "Country" and c != "2020-2025%"]
# Coerce numeric, Keep NaNs, and forward-fill per country across years
for c in year_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
# Forward-fill across years per row
df[year_cols] = df[year_cols].ffill(axis=1)
# Back-fill if leading NaNs
df[year_cols] = df[year_cols].bfill(axis=1)
# Melt to long/tidy format
tidy = df.melt(id_vars=["Country"], value_vars=year_cols,
               var_name="Year", value_name="GDP")
tidy["Year"] = tidy["Year"].astype(int)
tidy = tidy.sort_values(["Year", "GDP"],
                        ascending=[True, False]).reset_index(drop=True)
# Calculate per-year rank (1 = highest GDP)
tidy["Rank"] = tidy.groupby("Year")["GDP"].rank(method="first",
                                                ascending=False)


def pick_countries(n=10, by_year=2025, min_gdp=None, include=None):
    """
    Pick a set of countries to plot lines for.
    - n: top-N by GDP in by_year
    - min_gdp: minimum GDP threshold in by_year (overrides n if used)
    - include: list of countries to always include
    """
    year_df = tidy[tidy["Year"] == by_year].dropna(subset=["GDP"])
    chosen = set()
    if min_gdp is not None:
        chosen.update(year_df[year_df["GDP"] >= min_gdp]
                      ["Country"].head(196).list())
    else:
        chosen.update(year_df.sort_values("GDP", ascending=False)["Country"]
                      .head(n).tolist())
    if include:
        chosen.update(include)
    return sorted(chosen)


def plot_lines(countries=None, start=2020, end=2025):
    data = tidy[(tidy["Year"] >= start) & (tidy["Year"] <= end)]
    if countries is None:
        countries = pick_countries()
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        sub = data[data["Country"] == c].sort_values("Year")
        ax.plot(sub["Year"], sub["GDP"], label=c, linewidth=2)
    ax.set_title(f"GDP Evolution {start}-{end}")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("/Users/conallnoonan/Documents/GDP_Analysis/gdp_evolution.png")
    plt.close()


plot_lines()


def bar_chart_race(save_path="bar_chart_race.mp4", top_n=20, start=2020,
                   end=2025,
                   fps=0.8):
    years = list(range(start, end + 1))
    frames = []
    for y in years:
        frame = tidy[tidy["Year"] == y].nlargest(top_n, "GDP").copy()
        frame = frame.sort_values("GDP", ascending=True)

        frames.append(frame)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_label("GDP")
    ax.set_title(f"Top {top_n} GDP - {years[0]}")

    def update(i):
        frame = frames[i]
        ax.clear()
        ax.barh(frame["Country"], frame["GDP"])
        ax.set_label("GDP")
        ax.set_title(f"Top {top_n} GDP - {years[i]}")
        ax.grid(True, axis="x", alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(years), interval=1000//fps,
                         repeat=False)
    try:
        anim.save(save_path, fps=fps, dpi=150)
        print(f"Saved: {Path(save_path).resolve()}")
    except Exception as e:
        print("Could not save video (is ffmpeg installed)? Showing inline "
              "instead.", e)
    plt.show()
    plt.close()


bar_chart_race()
