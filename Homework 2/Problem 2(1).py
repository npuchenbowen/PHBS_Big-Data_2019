import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

df = pd.read_csv('climate_change_1.csv')

def closed_form_2():
    lamb = 0.5
    df['const'] = 1
    traindata = df[df.Year<=2006]
    testdata = df[df.Year>2006]
    X = traindata[['MEI','CO2','CH4','N2O','CFC-11','CFC-12','TSI','Aerosols','const']]
    Y = traindata['Temp']

    Theta2 = np.dot(
        np.dot(np.linalg.inv((np.dot(X.T, X) + lamb * np.eye(X.shape[1]))), X.T), Y)
    print(Theta2)
    
closed_form_2()
