import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from multicollinearity import multicollinearity_check

#USA
#SETUP DATA
gdp = pd.read_csv('./Data/USA_REAL_GDP.csv')
gdp.set_index('DATE', inplace=True)

income = pd.read_csv('./Data/USA_INCOME_GINI.csv')
income.set_index('DATE', inplace=True)

moneySupply = pd.read_csv('./Data/USA_MV.csv')
moneySupply.set_index('DATE', inplace=True)

inflation = pd.read_csv('./Data/Inflation Data (Long).csv')
inflation['DATE'] = pd.DatetimeIndex(inflation['DATE']).year
inflation.set_index('DATE', inplace=True)

usaTradeVol = pd.read_csv('./Data/USA_TRADE_VOLUME.csv')
usaTradeVol.set_index('DATE', inplace=True)

#MULTICOLLINEARITY
corrFrame = pd.DataFrame()
corrFrame['DATE'] = usaTradeVol.index
corrFrame.set_index('DATE', inplace=True)
corrFrame['Total Trade Volume'] = usaTradeVol[['TRADEVOL']].pct_change()
corrFrame['Income'] = income['gini'].values.flatten()
corrFrame['Money Velocity'] = moneySupply['MV'].values.flatten()
corrFrame.dropna(inplace=True)
print(corrFrame.corr())
multicollinearity_check(corrFrame)

#SETUP REGERESSION DATAFRAME
regressionFrame = pd.DataFrame()
flatInflation = inflation.values.flatten()[10:]
regressionFrame['Inflation'] = flatInflation
regressionFrame.index = usaTradeVol.index
regressionFrame['Change In Total Trade Volume'] = usaTradeVol[['TRADEVOL']].pct_change()
regressionFrame['Income Gini'] = income['gini'].values.flatten()
regressionFrame['Money Velocity'] = moneySupply['MV'].values.flatten()
regressionFrame.dropna(inplace=True)

# PERFORM REGRESSION
X = regressionFrame[['Change In Total Trade Volume', 'Income Gini', 'Money Velocity']]
y = regressionFrame['Inflation']
est = sm.OLS(y, X).fit()
print(est.summary())