import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

goodsAndServices = pd.read_csv('./Data/Goods And Services Data (Long).csv')
usaGoodsAndServices = goodsAndServices[(goodsAndServices['LOCATION'] == 'USA')]
usaGoodsAndServices = usaGoodsAndServices[usaGoodsAndServices['FREQUENCY'] == 'A']
usaGoodsAndServices = usaGoodsAndServices[usaGoodsAndServices['MEASURE'] == 'MLN_USD']
usaGoodsAndServices.rename(columns={'TIME': 'DATE'}, inplace=True)
usaGoodsAndServices.set_index('DATE', inplace=True)
usaGoodsAndServicesEXP = usaGoodsAndServices[usaGoodsAndServices['SUBJECT'] == 'EXP']
usaGoodsAndServicesIMP = usaGoodsAndServices[usaGoodsAndServices['SUBJECT'] == 'IMP']
usaGoodsAndServices = usaGoodsAndServicesEXP
usaGoodsAndServices.rename(columns={'Value' : 'Exports'}, inplace=True)
usaGoodsAndServices['Imports'] = usaGoodsAndServicesIMP[['Value']].values.flatten()
usaGoodsAndServices['Total Trade Volume'] = usaGoodsAndServices['Exports'] + usaGoodsAndServices['Imports']
usaGoodsAndServices = usaGoodsAndServices[['Total Trade Volume']]

# INFLATION
inflation = pd.read_csv('./Data/Inflation Data (Long).csv')
inflation['DATE'] = pd.DatetimeIndex(inflation['DATE']).year
inflation.set_index('DATE', inplace=True)

# CORRELATIONS (-.63)
flatInflation = inflation.values.flatten()
corrDf = pd.DataFrame()
corrDf['Total Trade Volume'] = usaGoodsAndServices['Total Trade Volume']
corrDf['Inflation'] = flatInflation[10:]
goodsAndServicesInflationCorr = corrDf.corr().values[0][1]
print(goodsAndServicesInflationCorr)

# PLOT TRADE
fig = plt.figure()
ax = plt.gca()
ax2 = ax.twinx()
ax.plot(usaGoodsAndServices.index, usaGoodsAndServices['Total Trade Volume'] / 1000, sns.xkcd_rgb["pale red"], label='US Total Trade Volume')
ax2.plot(inflation.index[10:], inflation['FPCPITOTLZGUSA'][10:], label='Inflation')
ax.set_title('US Trade Volume Compared With Inflation')
ax.set_xlabel('Year')
ax.set_ylabel('US Trade Volume (In Billions)')
ax2.set_ylabel('Inflation Rate (%)')
ax.legend()
ax2.legend()
plt.show()



