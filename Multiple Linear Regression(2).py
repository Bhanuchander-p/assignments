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


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


# Rearange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP", "WT","MPG"])
cars.head()


# In[4]:


cars.info()


# In[5]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})


# In[6]:


# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x label for the boxplot


# In[7]:


# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')


# In[8]:


# Adjust Layout
plt.tight_layout()
plt.show()


# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})


# In[10]:


# Creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x label for the boxplot


# In[11]:


# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})


# In[13]:


# Creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x label for the boxplot


# In[14]:


# Creating a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# ### Observations from boxplot and histogrms
# - There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# - In VOL and WT columns,a few outliers are observed in both tails of their distributions.
# - The extreme values of cars data may have come from the specially designed nature of cars.
# - As this is multi-dimensinal data, the outliers with respect to spatial dimensions may have to be considered while bulding the regression model

# In[11]:


# Checking for duplicated rows
cars[cars.duplicated()]


# #### Pair plots and Correlation Coefficients

# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# ### Observation from correlation plots and Coefficients
# - Between x and y, all the variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# In[13]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[14]:


model1.summary()


# ### Observations from model summary
# - The R-squared and adjusted R-suared values are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is close to zero, indicating that all or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[16]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[20]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))



# ### Check for multicollinearity among X-columns using VIF metho

# In[21]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### Observations for VIF values:
# - The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity problem.
# - Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity.
# - It is decided to drop WT and retian VOL column in further models

# In[23]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[27]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[28]:


model2.summary()


# ### Performance metrics form model2

# In[29]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[30]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[32]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# In[15]:


# DEfine variables and assign values
k = 3 # no of x-columns in cars1
n = 81 # no of observations (rows)
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[16]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=.05)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[36]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[38]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[44]:


model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[46]:


model3.summary()


# In[48]:


# Predict on all x data columns 
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[50]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse)
print("RSME :",np.sqrt(mse))


# In[ ]:




