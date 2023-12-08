import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

df_q1=pd.read_csv("C:\\Users\\kamal\\Desktop\\daily_covid_cases.csv")
original=df_q1['new_cases']

###Q1 part a
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df_q1['Date'],
           df_q1['new_cases'].values,
           color='purple')
ax.set(xlabel="Date", ylabel="new_cases",
       title="Q1 part a")
date_form = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation =45)
plt.show()

###Q1 part b
one_day_lag=original.shift(1)
print("Q1 part b --> Pearson correlation (autocorrelation) coefficient :",original.corr(one_day_lag))
print()

###Q1 part c
plt.scatter(original, one_day_lag, s=5)
plt.xlabel("Given time series data")
plt.ylabel("One day lagged time series data")
plt.title("Q1 part c")
plt.show()

###Q1 part d
PCC=sm.tsa.acf(original)
lag=[1,2,3,4,5,6]
pcc=PCC[1:7]
plt.plot(lag,pcc, marker='o')
for xitem,yitem in np.nditer([lag, pcc]):
        etiqueta = "{:.3f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center")
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part d")
plt.show()

###Q1 part e
plot_acf(x=original, lags=50)
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part e")
plt.show()