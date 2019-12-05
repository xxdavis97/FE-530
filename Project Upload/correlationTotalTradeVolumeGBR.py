import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

goodsAndServices = pd.read_csv('./Data/Goods And Services Data (Long).csv')
gbrGoodsAndServices = goodsAndServices[(goodsAndServices['LOCATION'] == 'GBR')]
gbrGoodsAndServices = gbrGoodsAndServices[gbrGoodsAndServices['FREQUENCY'] == 'A']
gbrGoodsAndServices = gbrGoodsAndServices[gbrGoodsAndServices['MEASURE'] == 'MLN_USD']
gbrGoodsAndServices.rename(columns={'TIME': 'DATE'}, inplace=True)
gbrGoodsAndServices.set_index('DATE', inplace=True)
gbrGoodsAndServicesEXP = gbrGoodsAndServices[gbrGoodsAndServices['SUBJECT'] == 'EXP']
gbrGoodsAndServicesIMP = gbrGoodsAndServices[gbrGoodsAndServices['SUBJECT'] == 'IMP']
gbrGoodsAndServices = gbrGoodsAndServicesEXP
gbrGoodsAndServices.rename(columns={'Value' : 'Exports'}, inplace=True)
gbrGoodsAndServices['Imports'] = gbrGoodsAndServicesIMP[['Value']].values.flatten()
gbrGoodsAndServices['Total Trade Volume'] = gbrGoodsAndServices['Exports'] + gbrGoodsAndServices['Imports']
gbrGoodsAndServices = gbrGoodsAndServices[['Total Trade Volume']]

# INFLATION
inflation = pd.read_csv('./Data/GBR Inflation Data (Long).csv')
inflation['DATE'] = pd.DatetimeIndex(inflation['DATE']).year
inflation.set_index('DATE', inplace=True)

# CORRELATIONS (-.657)
flatInflation = inflation.values.flatten()
corrDf = pd.DataFrame()
corrDf['Total Trade Volume'] = gbrGoodsAndServices['Total Trade Volume']
corrDf['Inflation'] = flatInflation[10:]
goodsAndServicesInflationCorr = corrDf.corr().values[0][1]
print(goodsAndServicesInflationCorr)

# PLOT TRADE
fig = plt.figure()
ax = plt.gca()
ax2 = ax.twinx()
ax.plot(gbrGoodsAndServices.index, gbrGoodsAndServices['Total Trade Volume'] / 1000, sns.xkcd_rgb["pale red"], label='Great Britain Total Trade Volume')
ax2.plot(inflation.index[10:], inflation['FPCPITOTLZGGBR'][10:], label='Inflation')
ax.set_title('British Trade Volume Compared With Inflation')
ax.set_xlabel('Year')
ax.set_ylabel('British Trade Volume (In Billions)')
ax2.set_ylabel('Inflation Rate (%)')
ax.legend()
ax2.legend()
plt.show()



