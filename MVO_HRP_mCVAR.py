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
startDate = dt.datetime(2021, 4, 6)

# Healthcare-Stocks
stockList_Healthcare = ['GEHC', 'PFE', 'JNJ']

# Tech-Stocks
stockList_Tech = ['GOOG', 'META', 'AAPL']

# Retail-Stocks
stockList_Retail = ['COST', 'WMT', 'KR']

# Finance-Stocks
stockList_Finance = ['JPM', 'BAC', 'HSBC']

# More Stocks
# stockList_Rest = ['HUM', 'AIXI', 'YM=F']

portfolio_List = stockList_Healthcare + stockList_Tech + stockList_Retail + stockList_Finance #+ stockList_Rest
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


fig, ax = plt.subplots(figsize=(12, 8))

#MVO
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)
# Mark Max Sharpe
ret_sharpe, std_sharpe, _ = ef_sharpe.portfolio_performance()
ax.scatter(std_sharpe, ret_sharpe, marker="*", s=200, c="red", zorder=5, label="Max Sharpe")

risk_free_rate = 0.05  # ~current US T-bill rate
slope = (ret_sharpe - risk_free_rate) / std_sharpe
x_cml = [0, std_sharpe * 2 ]
y_cml = [risk_free_rate, risk_free_rate + slope * std_sharpe * 2]
ax.plot(x_cml, y_cml, linestyle='--', color='orange', label='Capital Market Line')


# Mark Min Volatility
ef_minvol = EfficientFrontier(mu, S)
ef_minvol.min_volatility()
ret_minvol, std_minvol, _ = ef_minvol.portfolio_performance()
ax.scatter(std_minvol, ret_minvol, marker="*", s=200, c="green", zorder=5, label="Min Volatility")

# --- Label each individual asset ---
tickers = portfolio.columns.tolist()
asset_returns = mu.values                        # Expected return per asset
asset_vols = np.sqrt(np.diag(S.values))          # Volatility = sqrt of variance (diagonal of cov matrix)




sector_colors = {
    'GEHC': 'red', 'PFE': 'red', 'JNJ': 'red',       # Healthcare
    'GOOG': 'blue', 'META': 'blue', 'AAPL': 'blue',   # Tech
    'COST': 'green', 'WMT': 'green', 'KR': 'green',   # Retail
    'JPM': 'purple', 'BAC': 'purple', 'HSBC': 'purple' # Finance
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
ax.scatter(hrp_vol, hrp_ret, marker="D", s=100, c="cyan", zorder=5, label=f"HRP (Sharpe: {hrp_sharpe:.2f})")

# # Plot where mCVAR portfolio lands
# ef_cvar2 = EfficientCVaR(mu, returns)
# ef_cvar2.min_cvar()

# # EfficientCVaR returns (expected_return, cvar) — no Sharpe
# cvar_ret, cvar_cvar = ef_cvar.portfolio_performance()


# ax.scatter(cvar_cvar, cvar_ret, marker="D", s=100, c="magenta", zorder=5, label=f"mCVAR (return: {cvar_ret:.2f})")

ax.legend()
plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150)
plt.show()

print('The key takeaway: ' \
    'HRP being this close to the frontier while being far more robust and ' \
    'diversified makes it arguably the best practical choice among your three methods for this portfolio.')

print('')