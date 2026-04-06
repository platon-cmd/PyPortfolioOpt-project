import pandas as pd
import yfinance as yf
import datetime as dt

#import data
def get_Data (stocks, start, end):
    stocks_Data = yf.download(stocks, start, end)
    stocks_Data = stocks_Data['Close']
    return stocks_Data
    # returns = stocks_Data.pct_change()
    # meanReturns = returns.mean()
    # covMatrix = returns.cov()
    # return meanReturns, covMatrix


endDate = dt.datetime.now()
startDate = dt.datetime(2021, 4, 6)

# Healthcare-Stocks
stockList_Healthcare = ['MRNA', 'PFE', 'JNJ']
# stocks_Healthcare = [stock_HC for stock_HC in stockList_Healthcare]
# get_Data(stocks_Healthcare, startDate, endDate)

# Tech-Stocks
stockList_Tech = ['GOOG', 'META', 'AAPL']
# stocks_Tech = [stock_T for stock_T in stockList_Tech]
# get_Data(stocks_Tech, startDate, endDate)

# Retail-Stocks
stockList_Retail = ['COST', 'WMT', 'KR']
# stocks_Retail = [stock_R for stock_R in stockList_Retail]
# get_Data(stocks_Retail, startDate, endDate)

# Finance-Stocks
stockList_Finance = ['JPM', 'BAC', 'HSBC']
# stocks_Finance = [stock_F for stock_F in stockList_Finance]
# get_Data(stocks_Finance, startDate, endDate)

portfolio_List = stockList_Healthcare + stockList_Tech + stockList_Retail + stockList_Finance
print(f'The portfolio consists of the following stocks: {portfolio_List}')
stock_Portfolio = [stock for stock in portfolio_List]
portfolio = get_Data(stock_Portfolio, startDate, endDate)
print(portfolio.head())
portfolio.to_csv('portfolio.csv', index = False)
portfolio = pd.read_csv('portfolio.csv')