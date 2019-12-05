getSymbols('M2', from = '1981-01-01', to = '2019-01-01', src = 'FRED')

money_supply = M2   #weekly data

money_supply = to.quarterly(money_supply)

money_supply = money_supply$money_supply.Close

money_supply
tail(CPI)
length(money_supply)

money_supply = money_supply[1:154]
tail(money_supply)

adjusted_CPI = CPI[136:289]

cor(adjusted_CPI, money_supply)

money_growth = quarterlyReturn(money_supply)

