# Feature_Transformation
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM:
NAME:Pooja V

REG NO: 212221040122
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:
![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/7ae3f269-1c48-4066-bc77-389f69b99e29)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/98907d0a-0468-4084-9ef3-dd052b2922b3)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/93ce585f-f4c5-40fe-9c1f-ef1a31e03372)

![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/11d110c2-5aff-4131-9956-8fe43181cf8c)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/56c63ece-394d-406d-8dc2-bb63b2640d91)
![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/7aa26b6a-147f-4534-bc9a-b0cb3b8623bc)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/e75a7bfb-e11b-4d87-92df-70e4d83a5317)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/247416ca-5cba-47a5-b904-ca4036bccf19)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/b93973d0-fed8-4d5f-bb09-1501890d42e4)

![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/7144c126-24cf-4f68-bc1a-c5ad37ef62c1)


![image](https://github.com/Poojariyaa/ODD2023-Datascience-Ex06/assets/127511817/7e7693fc-8e1d-4844-aaea-98c0630408d2)

# RESULT:
Thus feature transformation is done for the given dataset.
