library('quantmod')
library('xts')
getSymbols('CPIAUCSL', from = '1947-01-01', to = '2019-01-01', src = 'FRED')

CPI = CPIAUCSL

getSymbols('GDP', from = '1947-01-01', to = '2019-01-01', src = 'FRED')

CPI = CPI[1:865]

CPI = to.quarterly(CPI)

CPI = CPI$CPI.Close

GDP = to.quarterly(GDP)

GDP = GDP$GDP.Close

GDP = GDP[1:289]

plot(GDP, main = 'GDP Over Time')

plot(CPI, main = 'CPI Over Time')

cor(CPI, GDP)

CPI_growth = quarterlyReturn(CPI)
GDP_growth = quarterlyReturn(GDP)

plot(GDP_growth, main = 'GDP Growth Over Time')

plot(CPI_growth, main = 'Inflation Over Time')
cor(GDP_growth, CPI_growth)

yearly_GDP = to.yearly(GDP)
yearly_GDP = yearly_GDP$GDP.Close

yearly_GDP = annualReturn(yearly_GDP)
yearly_GDP = yearly_GDP[1:72]

plot(yearly_GDP)
mean(yearly_GDP)   #long term average GDP growth
sd(yearly_GDP)     #long term Standard deviation
mean(yearly_GDP[65:72])    #short term aferage GDP growth

CPI_annual = to.yearly(CPI)
CPI_annual = CPI_annual$CPI.Close

annual_inflation = annualReturn(CPI_annual)
annual_inflation = annual_inflation[1:72]

cor(annual_inflation, yearly_GDP)
mean(annual_inflation)
mean(annual_inflation[65:72])
sd(annual_inflation[65:72])

plot(yearly_GDP, main = 'US GDP Growth Rate Over Time')
plot(annual_inflation, main = 'US Inflation Rate Over Time')

cor(annual_inflation, yearly_GDP)
yearly_GDP
