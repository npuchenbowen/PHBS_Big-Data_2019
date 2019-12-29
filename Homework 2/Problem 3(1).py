import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

df = pd.read_csv('climate_change_1.csv')
climate_change_1_train = df[df.Year<=2006]
climate_change_1_test = df[df.Year>2006]

climate_change_1_train_X = climate_change_1_train.iloc[:,2:10]
climate_change_1_train_Y = climate_change_1_train.iloc[:,10]

print(climate_change_1_train_X.corr())
#80% above is considered as highly related
#CO2 CH4 N2O CFC-12 are highly correlated with each other
#CFC-11 CFC-12 are correlated with each other
 
#delete N2O,CH4,CFC-12
climate_change_1_train_X_new = climate_change_1_train.loc[:,['MEI','CO2','CFC-11','TSI','Aerosols']]

# 1.2 run a regression remove not significant with alpha = 0.01
import statsmodels.api as sm
from statsmodels import regression
climate_change_1_train_X_new=sm.add_constant(climate_change_1_train_X_new)
model=sm.OLS(climate_change_1_train_Y,climate_change_1_train_X_new)
res=model.fit()
print(res.summary())

#delete CFC-11 which is not significant
 
# 1.3 run the final model
climate_change_1_train_X_new_2 = climate_change_1_train.loc[:,['MEI','CO2','TSI','Aerosols']]
climate_change_1_train_X_new_2=sm.add_constant(climate_change_1_train_X_new_2)
model=sm.OLS(climate_change_1_train_Y,climate_change_1_train_X_new_2)
res=model.fit()
print(res.summary())