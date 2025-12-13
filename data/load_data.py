import pandas as pd
from ucimlrepo import fetch_ucirepo

stocks = fetch_ucirepo(id=390)

print(stocks.metadata)
print(stocks.variables)