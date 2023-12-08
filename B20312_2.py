import pandas as pd 
import matplotlib.pyplot as plt 
import operator
import numpy as np 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR
series = pd.read_csv("C:\\Users\\kamal\\Desktop\\daily_covid_cases.csv",parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65 # 35% for testing
X = series.values
train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

window = 5 # The lag=1
model = AR(train, lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print()
print("Q2 part a--> coefficients are :",coef)
print()
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

### Q2 part b,,, part 1
plt.scatter(test,predictions )
plt.xlabel('Actual cases')
plt.ylabel('Predicted cases')
plt.title('Q2 part b\n Part 1')
plt.show()

### Q2 part b,,, part 2
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases')
plt.plot(x,predictions , label='Predicted cases')
plt.legend()
plt.title('Q2 part b\n Part 2')
plt.show()

### Q2 part b,,, part 3
rmse=mean_squared_error(test, predictions,squared=False)
print("Q2 part b-1--> persent RMSE :",rmse*100/(sum(test)/len(test)),"%")
print()

mape=mean_absolute_percentage_error(test, predictions)
print("Q2 part b-1--> persent MAPE :",mape)
