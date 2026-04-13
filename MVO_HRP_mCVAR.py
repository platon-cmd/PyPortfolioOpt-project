import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np 
from adjustText import adjust_text

# For MVO
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# For HRP
from pypfopt import HRPOpt

#For mCVAR
from pypfopt.efficient_frontier import EfficientCVaR

# Plotting
from pypfopt import plotting
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

#import data
def get_Data (stocks, start, end):
    stocks_Data = yf.download(stocks, start, end)
    stocks_Data = stocks_Data['Close']
    return stocks_Data


endDate = dt.datetime.now()
startDate = dt.datetime(2018, 4, 9)

# Healthcare-Stocks
stockList_Healthcare = ['GEHC', 'PFE', 'JNJ']

# Tech-Stocks
stockList_Tech = ['GOOG', 'META', 'AAPL']

# Retail-Stocks
stockList_Retail = ['COST', 'WMT', 'KR']

# Finance-Stocks
stockList_Finance = ['JPM', 'BAC', 'HSBC']

# ETF
stock_ETF = ['^GSPC', 'SOXX', 'SPY']
# stockList_Rest = ['HUM', 'AIXI', 'YM=F']

portfolio_List = stockList_Healthcare + stockList_Tech + stockList_Retail + stockList_Finance + stock_ETF #+ stockList_Rest
print(f'The portfolio consists of the following stocks: {portfolio_List}')
stock_Portfolio = [stock for stock in portfolio_List]
portfolio = get_Data(stock_Portfolio, startDate, endDate)
print(portfolio.head())
# portfolio.to_csv('portfolio.csv', index = False)
# portfolio = pd.read_csv('portfolio.csv')

# Calculate the covariance matrix and store the calculated returns in variables S and mu, respectively:
mu = mean_historical_return(portfolio)
S = CovarianceShrinkage(portfolio).ledoit_wolf()

# Next, calculate the weights. Here, we will use the max Sharpe statistic. 
# The Sharpe ratio is the ratio between returns and risk. 
# The lower the risk and the higher the returns, the higher the Sharpe ratio. 
# The algorithm looks for the maximum Sharpe ratio, which translates to the portfolio with the highest return and lowest risk.
# Ultimately, the higher the Sharpe ratio, the better the performance of the portfolio. 
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

# We can also display portfolio performance:
ef.portfolio_performance(verbose=True)

# Finally, let’s convert the weights into actual allocations values (i.e., how many of each stock to buy). 
# For our allocation, let’s consider an investment amount of $100,000:
latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=1000)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))


# --- Efficient Frontier Plot with Asset Labels --- MVO
ef_plot = EfficientFrontier(mu, S)
ef_sharpe = EfficientFrontier(mu, S)
ef_sharpe.max_sharpe()


# Mean variance optimization doesn’t perform very well since it makes many simplifying assumptions, 
# such as returns being normally distributed and the need for an invertible covariance matrix.
# Fortunately, methods like HRP and mCVAR address these limitations. 

# Calculate the returns:
print('\n')
returns = portfolio.pct_change().dropna()

# Run the optimization algorithm to get the weights:
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()

# Print the performance of the portfolio and the weights:
hrp.portfolio_performance(verbose=True)
print(dict(hrp_weights))

# Calculate the discrete allocation using our weights:
da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=1000)

allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}".format(leftover))




# The mCVAR is another popular alternative to mean variance optimization. 
# It works by measuring the worst-case scenarios for each asset in the portfolio, which is represented here by losing the most money. 
# The worst-case loss for each asset is then used to calculate weights to be used for allocation for each asset. 

# Convert prices to returns
print('\n')
returns = portfolio.pct_change().dropna()

ef_cvar = EfficientCVaR(mu, returns)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
print(dict(cleaned_weights))

da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=1000)

allocation, leftover = da_cvar.greedy_portfolio()
print("Discrete allocation (CVAR):", allocation)
print("Funds remaining (CVAR): ${:.2f}".format(leftover))



# --- Monte Carlo Simulation ---
num_portfolios = 5000
mc_returns = np.zeros(num_portfolios)
mc_volatilities = np.zeros(num_portfolios)
mc_sharpes = np.zeros(num_portfolios)
mc_weights_list = np.zeros((num_portfolios, len(portfolio_List)))

np.random.seed(42)
for i in range(num_portfolios):
    # Random weights that sum to 1
    w = np.random.dirichlet(np.ones(len(portfolio_List)))
    mc_weights_list[i] = w

    port_return = np.dot(w, mu)
    port_vol = np.sqrt(w.T @ S.values @ w)
    sharpe = (port_return - 0.05) / port_vol  # same risk-free rate as your CML

    mc_returns[i] = port_return
    mc_volatilities[i] = port_vol
    mc_sharpes[i] = sharpe



fig, (ax, ax_weights) = plt.subplots(
    2, 1, figsize=(10, 8),  # was (14, 12)
    gridspec_kw={'height_ratios': [3, 1]}
)

# Draw the Monte Carlo cloud
sc = ax.scatter(
    mc_volatilities, mc_returns,
    c=mc_sharpes,
    cmap="viridis",
    alpha=0.3,
    s=10,
    zorder=1,
    label="_nolegend_"
)
plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

#MVO
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)
# Mark Max Sharpe
ret_sharpe, std_sharpe, _ = ef_sharpe.portfolio_performance()
ax.scatter(std_sharpe, ret_sharpe, marker="X", s=100, c="red", zorder=5, label=f"MVO (Sharpe {round(ret_sharpe, 2)})")

risk_free_rate = 0.05  # ~current US T-bill rate
slope = (ret_sharpe - risk_free_rate) / std_sharpe
x_cml = [0, std_sharpe * 2 ]
y_cml = [risk_free_rate, risk_free_rate + slope * std_sharpe * 2]
ax.plot(x_cml, y_cml, linestyle='--', color='orange', label='Capital Market Line')


# Mark Min Volatility
ef_minvol = EfficientFrontier(mu, S)
ef_minvol.min_volatility()
ret_minvol, std_minvol, _ = ef_minvol.portfolio_performance()
ax.scatter(std_minvol, ret_minvol, marker="X", s=100, c="green", zorder=5, label="Min Volatility")

# --- Label each individual asset ---
tickers = portfolio.columns.tolist()
asset_returns = mu.values                        # Expected return per asset
asset_vols = np.sqrt(np.diag(S.values))          # Volatility = sqrt of variance (diagonal of cov matrix)

sector_colors = {
    'GEHC': 'red', 'PFE': 'red', 'JNJ': 'red',       # Healthcare
    'GOOG': 'blue', 'META': 'blue', 'AAPL': 'blue',   # Tech
    'COST': 'green', 'WMT': 'green', 'KR': 'green',   # Retail
    'JPM': 'purple', 'BAC': 'purple', 'HSBC': 'purple', # Finance
    '^GSPC': 'burlywood', 'SOXX': 'burlywood', 'SPY': 'burlywood' # ETFs
    }

texts = []
for ticker in portfolio_List:
    idx = portfolio_List.index(ticker)
    ax.scatter(asset_vols[idx], asset_returns[idx],
               color=sector_colors[ticker], s=60, zorder=4)
    texts.append(ax.text(asset_vols[idx], asset_returns[idx],
                         ticker, fontsize=9, fontweight="bold",
                         color=sector_colors[ticker]))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    

ax.set_title("Efficient Frontier with Asset Labels", fontsize=14)
ax.set_xlabel("Annual Volatility (Risk)")
ax.set_ylabel("Annual Expected Return")


# Plot where HRP portfolio lands on the risk/return space
hrp_ret, hrp_vol, hrp_sharpe = hrp.portfolio_performance()
ax.scatter(hrp_vol, hrp_ret, marker="X", s=100, c="cyan", zorder=5, label=f"HRP (Sharpe: {hrp_sharpe:.2f})")

# # Plot where mCVAR portfolio lands
# ef_cvar2 = EfficientCVaR(mu, returns)
# ef_cvar2.min_cvar()

# # EfficientCVaR returns (expected_return, cvar) — no Sharpe
# cvar_ret, cvar_cvar = ef_cvar.portfolio_performance()


# ax.scatter(cvar_cvar, cvar_ret, marker="D", s=100, c="magenta", zorder=5, label=f"mCVAR (return: {cvar_ret:.2f})")

ax.legend()
plt.tight_layout()

# --- Collect weights for Max Sharpe and HRP ---
ef_ws = EfficientFrontier(mu, S)
ef_ws.max_sharpe()
mvo_weights_clean = ef_ws.clean_weights()

mvo_vals = [mvo_weights_clean.get(t, 0) for t in portfolio_List]
hrp_vals = [hrp_weights.get(t, 0) for t in portfolio_List]

x = np.arange(len(portfolio_List))
bar_width = 0.35

bars_mvo = ax_weights.bar(x - bar_width/2, mvo_vals, bar_width,
                           label='Max Sharpe (MVO)',
                           color=[sector_colors[t] for t in portfolio_List],
                           alpha=0.9, edgecolor='black', linewidth=0.5)

bars_hrp = ax_weights.bar(x + bar_width/2, hrp_vals, bar_width,
                           label='HRP',
                           color=[sector_colors[t] for t in portfolio_List],
                           alpha=0.45, edgecolor='black', linewidth=0.5,
                           hatch='//')

# Add percentage labels on top of each bar
for bar in bars_mvo:
    h = bar.get_height()
    if h > 0.01:  # only label if > 1% to avoid clutter
        ax_weights.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.0%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

for bar in bars_hrp:
    h = bar.get_height()
    if h > 0.01:
        ax_weights.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.0%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax_weights.set_xticks(x)
ax_weights.set_xticklabels(portfolio_List, fontsize=9)
ax_weights.set_ylabel("Portfolio Weight")
ax_weights.set_title("Stock Weights: Max Sharpe vs HRP", fontsize=11)
ax_weights.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax_weights.legend()
ax_weights.set_ylim(0, max(max(mvo_vals), max(hrp_vals)) + 0.08)

# Sector dividers — vertical lines between sector groups
for boundary in [2.5, 5.5, 8.5]:  # after Healthcare, Tech, Retail
    ax_weights.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Sector labels along x-axis
for label, pos in [('Healthcare', 1), ('Tech', 4), ('Retail', 7), ('Finance', 10), ('ETF', 13)]:
    ax_weights.text(pos, ax_weights.get_ylim()[1] * 0.95,
                    label, ha='center', fontsize=8, color='gray', style='italic')
plt.savefig("efficient_frontier.png", dpi=150)
plt.show(block=False)   # ← Fenster öffnet sich, Code läuft sofort weiter
plt.pause(0.1)

print('The key takeaway: ' \
    'HRP being this close to the frontier while being far more robust and ' \
    'diversified makes it arguably the best practical choice among your three' \
    ' methods for this portfolio.')


# ═══════════════════════════════════════════════════════════════════════════════
# ---------------ROLLING BACKTEST -----------------
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.dates as mdates

def rolling_backtest(
    prices: pd.DataFrame,
    lookback_months: int = 24,
    rebalance_freq: str = "QS",
    initial_value: float = 1000,
    risk_free_rate: float = 0.05,
):
    rebalance_dates = pd.date_range(
        start=prices.index[0] + pd.DateOffset(months=lookback_months),
        end=prices.index[-1],
        freq=rebalance_freq,
    )

    def snap(d):
        future = prices.index[prices.index >= d]
        return future[0] if len(future) else None

    rebalance_dates = sorted(set(
        snap(d) for d in rebalance_dates if snap(d) is not None
    ))

    portfolio_values = {"MVO": pd.Series(dtype=float),
                        "HRP": pd.Series(dtype=float),
                        "Equal Weight": pd.Series(dtype=float)}
    weight_history   = {"MVO": [], "HRP": []}
    cash             = {k: initial_value for k in portfolio_values}

    # Initialwert setzen 
    for k in portfolio_values:
        portfolio_values[k] = pd.Series(
            [initial_value],
            index=[prices.index[0]]
        )

    def get_weights_mvo(pw):
        try:
            mu_w = mean_historical_return(pw)
            S_w  = CovarianceShrinkage(pw).ledoit_wolf()
            ef   = EfficientFrontier(mu_w, S_w)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = ef.clean_weights()
            return np.array([w.get(t, 0) for t in pw.columns])
        except Exception:
            return np.ones(pw.shape[1]) / pw.shape[1]

    def get_weights_hrp(pw):
        ret_w = pw.pct_change().dropna()
        ret_w = ret_w.dropna(axis=1)
        ret_w = ret_w.loc[:, ret_w.std() > 1e-8]
        if ret_w.shape[1] < 2:
            return np.ones(pw.shape[1]) / pw.shape[1]
        hrp = HRPOpt(ret_w)
        w   = hrp.optimize()
        return np.array([w.get(t, 0) for t in pw.columns])

    for i, reb_date in enumerate(rebalance_dates):

        # nächstes Rebalancing-Datum 
        if i + 1 < len(rebalance_dates):
            next_reb = rebalance_dates[i + 1] - pd.Timedelta(days=1)
        else:
            next_reb = prices.index[-1]

        # sicherstellen, dass Zeitraum existiert
        if next_reb <= reb_date:
            continue

        # Lookback Window
        window_start = reb_date - pd.DateOffset(months=lookback_months)
        pw = prices.loc[window_start:reb_date]

        if len(pw) < 60:   
            continue

        # Gewichte berechnen 
        w_mvo = get_weights_mvo(pw)
        w_hrp = get_weights_hrp(pw)
        w_eq  = np.ones(pw.shape[1]) / pw.shape[1]

        # Gewichte IMMER normalisieren
        w_mvo = w_mvo / np.sum(w_mvo)
        w_hrp = w_hrp / np.sum(w_hrp)

        weight_history["MVO"].append(pd.Series(w_mvo, index=prices.columns, name=reb_date))
        weight_history["HRP"].append(pd.Series(w_hrp, index=prices.columns, name=reb_date))

        # --- Out-of-sample Zeitraum ---
        oos = prices.loc[reb_date:next_reb]

        if len(oos) < 2:
            continue

        oos = oos.iloc[1:]

        # --- Portfolio Entwicklung ---
        for strategy, w in [("MVO", w_mvo), ("HRP", w_hrp), ("Equal Weight", w_eq)]:

            returns = oos.pct_change().dropna()

            if returns.empty:
                continue

            # 👉 Safety: weights normalisieren
            w = w / np.sum(w)

            port_ret = returns @ w

            day_vals = pd.Series(
                cash[strategy] * (1 + port_ret).cumprod(),
                index=port_ret.index
            )

            cash[strategy] = day_vals.iloc[-1]

            portfolio_values[strategy] = pd.concat([
                portfolio_values[strategy],
                day_vals
            ])
            

    for k in weight_history:
        if weight_history[k]:
            weight_history[k] = pd.DataFrame(weight_history[k])

    return portfolio_values, weight_history


def calc_metrics(series, rf=0.05):
    dr       = series.pct_change().dropna()
    total    = series.iloc[-1] / series.iloc[0] - 1
    ann_ret  = (1 + total) ** (252 / len(series)) - 1
    ann_vol  = dr.std() * np.sqrt(252)
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd   = ((series / series.cummax()) - 1).min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return {"Total Return": f"{total:.1%}", "Ann. Return": f"{ann_ret:.1%}",
            "Ann. Volatility": f"{ann_vol:.1%}", "Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1%}", "Calmar": f"{calmar:.2f}"}


# ── Run ────────────────────────────────────────────────────────────────────────
print("\nRunning rolling backtest …")
results, wt_history = rolling_backtest(portfolio, lookback_months=12, rebalance_freq="QS")
print("Done.\n")

metrics_df = pd.DataFrame({k: calc_metrics(v) for k, v in results.items()})
print("── Out-of-Sample Metrics ────────────────────────────────────────")
print(metrics_df.to_string())




# ── Plot  (komplett neue, separate Figur — deine ursprüngliche bleibt erhalten)
colors_s = {"MVO": "crimson", "HRP": "steelblue", "Equal Weight": "gray"}

fig2, (ax_eq, ax_dd, ax_wt) = plt.subplots(
    3, 1, figsize=(12, 13),
    gridspec_kw={"height_ratios": [3, 2, 2]}
)
fig2.suptitle("Rolling Backtest (Out-of-Sample)", fontsize=15, fontweight="bold", y=1.01)

# — Panel 1: Equity curves
for name, series in results.items():
    ax_eq.plot(series.index, series, label=name,
               color=colors_s[name], linewidth=1.8)

for rd in pd.date_range(portfolio.index[0] + pd.DateOffset(months=12),
                        portfolio.index[-1], freq="QS"):
    ax_eq.axvline(rd, color="gray", alpha=0.2, linewidth=0.8, linestyle="--")

ax_eq.set_ylabel("Portfolio Value ($)")
ax_eq.set_title("Equity Curves", fontsize=12)
ax_eq.legend()
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_eq.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_eq.grid(alpha=0.3)

# — Panel 2: Drawdown
for name, series in results.items():
    dd = (series / series.cummax() - 1) * 100
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors_s[name])
    ax_dd.plot(dd.index, dd.values, color=colors_s[name], linewidth=1.0, label=name)

ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown", fontsize=12)
ax_dd.legend()
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_dd.grid(alpha=0.3)

# — Panel 3: MVO Gewichtsdrift
if isinstance(wt_history["MVO"], pd.DataFrame) and not wt_history["MVO"].empty:
    wdf    = wt_history["MVO"]
    bottom = np.zeros(len(wdf))
    for ticker in wdf.columns:
        ax_wt.bar(range(len(wdf)), wdf[ticker].values, bottom=bottom,
                  label=ticker, color=sector_colors.get(ticker, "gray"),
                  alpha=0.85, edgecolor="white", linewidth=0.4)
        bottom += wdf[ticker].values
    ax_wt.set_xticks(range(len(wdf)))
    ax_wt.set_xticklabels([d.strftime("%Y-%m") for d in wdf.index],
                           fontsize=8, rotation=30, ha="right")
    ax_wt.set_ylabel("Portfolio Weight")
    ax_wt.set_title("MVO Gewichtsdrift pro Quartal", fontsize=12)
    ax_wt.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_wt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7.5, ncol=1)
    ax_wt.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("rolling_backtest.png", dpi=150, bbox_inches="tight")
plt.show()