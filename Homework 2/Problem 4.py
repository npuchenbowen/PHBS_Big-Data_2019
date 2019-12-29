import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.preprocessing import normalize

df = pd.read_csv('climate_change_1.csv')

def Gradient_Descent(alpha, theta_0, x, y, tol):
    theta = theta_0
    cost = (1./(2*len(x))) * (np.sum((x @ theta_0 - y)**2))
    count = 0
    while not ((cost <= tol) | (count > 10000)):
        count += 1
        theta = theta - alpha * (1./len(x)) * (x.T @ (x @ theta - y))
        cost = (1./(2*len(x))) * (np.sum((x @ theta_0 - y)**2))
        print(theta)
    print('迭代共{}次'.format(count))
    print(theta)

def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        if c != 'const':
            d = df[c]
            MAX = d.max()
            MIN = d.min()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
        else:
            newDataFrame[c] = 1
    return newDataFrame
  
def closed_form_2():
    lamb = 0.1
    df['const'] = 1
    traindata = df[df.Year<=2006]
    testdata = df[df.Year>2006]
    X = traindata[['MEI','CO2','CH4','N2O','CFC-11','CFC-12','TSI','Aerosols','const']]
    Y = traindata['Temp']
    X = regularit(X)
    Y = (Y-Y.mean())/(Y.max()-Y.min())
    print(X)
    print(Y)
    
    Theta2 = [1,1,1,1,1,1,1,1,1]

    Gradient_Descent(0.001, Theta2, X, Y, 10)

closed_form_2()
