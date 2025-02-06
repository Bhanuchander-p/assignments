#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


caes= pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MP6"])
cars.head()


# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:


### Observation from boxplot and histograms
. There are some extreme values (outliers) observed in towards the right tail of sp and Hp distribution
. in VOL and WT columns, a few outliers are observed in both tails of there distribution
. The extreme values of cars data may have come from the special design nature of cars
. As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be comsidered


# In[ ]:


cars[cars.duplicated()]


# ### pair plots and Correlation Coefficients

# In[ ]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# ### Observations from correlation plots and Coeffcients
# . Between x and y, all the x variables as showing moderate to high correlation strengths, highest being between HP and MPG
# . Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# . Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
# . The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# ### preparing a preliminary model considering all X columns

# In[ ]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# ### Observation from model summary
# . The R-squared and adjusted R-suared values aare good and above 75% of variables in Y is explained by X columns
# . The probability values with respect to F-statistic is close to zero, indicating that all or some of x columns are significant
# . The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# ### performance metrics for model1

# In[7]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[ ]:




