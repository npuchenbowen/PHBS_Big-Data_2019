import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

df = pd.read_csv('climate_change_1.csv')

def R_square(Y_pred, Y):
     ESS = np.sum((Y_pred - Y.mean())**2)
     TSS = np.sum((Y - Y.mean())**2)
     R2 = ESS / TSS
     print(R2)
    

def closed_form_2():
    lamb = 0.1
    df['const'] = 1
    traindata = df[df.Year<=2006]
    testdata = df[df.Year>2006]
    X = traindata[['MEI','CO2','TSI','Aerosols',’const’]]
    Y = traindata['Temp']

    Theta2 = np.dot(
        np.dot(np.linalg.inv((np.dot(X.T, X) + lamb * np.eye(X.shape[1]))), X.T), Y)
    print(Theta2)
    
    Y_pred = np.dot(X,Theta2)
    R_square(Y_pred,Y)
    
closed_form_2()
