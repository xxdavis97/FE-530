library('tframePlus')
getSymbols('MKTGDPGBA646NWDB', from = '1960-01-01', to = '2018-01-01', src = 'FRED')

UK_GDP = MKTGDPGBA646NWDB  #UK GDP levels by year

UK_GDP

UK_GDP_change = annualReturn(UK_GDP)
mean(UK_GDP_change)
sd(UK_GDP_change)

plot(UK_GDP_change, main = 'UK Growth Rate Over Time')

length(UK_GDP_change)


mean(UK_GDP_change[52:59])
sd(UK_GDP_change[52:59])

getSymbols('GBRCPIALLMINMEI', from = '1960-01-01', to = '2018-01-01', src = 'FRED')

UK_CPI = GBRCPIALLMINMEI

plot(UK_CPI)
UK_CPI

UK_CPI = to.yearly(UK_CPI, indexAt = 'startof')
UK_CPI

UK_CPI = UK_CPI$UK_CPI.Close
length(UK_CPI)

UK_inflation = annualReturn(UK_CPI)

plot(UK_inflation, main = 'UK Inflation Rate Over Time')
mean(UK_inflation)
sd(UK_inflation)
length(UK_inflation)
mean(UK_inflation[52:60])
sd(UK_inflation[52:60])

adjust_inflation = UK_inflation[1:59]

cor(adjust_inflation, UK_GDP_change)
