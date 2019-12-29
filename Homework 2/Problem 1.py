import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

df = pd.read_csv('climate_change_1.csv')

def closed_form_1():
    traindata = df[df.Year<=2006]
    testdata = df[df.Year>2006]
    X = traindata[['MEI','CO2','CH4','N2O','CFC-11','CFC-12','TSI','Aerosols']]
    Y = traindata['Temp']
    est = sm.OLS(Y, sm.add_constant(X)).fit()
    print(est.summary())
    print(est.params)
    
closed_form_1()
