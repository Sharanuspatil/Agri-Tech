#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[24]:


df = pd.read_csv("/Users/Shreyu/Desktop/drone_agri/crop prediction/crop_production.csv")
df[:5]


# # Data Exploration

# In[25]:


df.isnull().sum()


# In[26]:


# Droping Nan Values
data = df.dropna()
print(data.shape)
test = df[~df["Production"].notna()].drop("Production",axis=1)
print(test.shape)


# In[29]:


data


# In[30]:


test


# In[31]:


for i in data.columns:
    print("column name :",i)
    print("No. of column :",len(data[i].unique()))
    print(data[i].unique())


# In[32]:


sum_maxp = data["Production"].sum()
data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)


# In[33]:


sum_maxp


# In[35]:


data[:5]


# # Data Visulization

# In[14]:


sns.lineplot(data["Crop_Year"],data["Production"])


# In[15]:


plt.figure(figsize=(25,10))
sns.barplot(data["State_Name"],data["Production"])
plt.xticks(rotation=90)


# In[16]:


sns.jointplot(data["Area"],data["Production"],kind='reg')


# In[17]:


sns.barplot(data["Season"],data["Production"])


# In[18]:


data.groupby("Season",axis=0).agg({"Production":np.sum})


# In[19]:


data["Crop"].value_counts()[:5]


# In[20]:


top_crop_pro = data.groupby("Crop")["Production"].sum().reset_index().sort_values(by='Production',ascending=False)
top_crop_pro[:5]


# ## Each type of crops required various area & various season. so, I'm going to pic top crop from this data

# ### 1.Rice

# In[21]:


rice_df = data[data["Crop"]=="Rice"]
print(rice_df.shape)
rice_df[:3]


# In[22]:


sns.barplot("Season","Production",data=rice_df)


# In[23]:


plt.figure(figsize=(13,10))
sns.barplot("State_Name","Production",data=rice_df)
plt.xticks(rotation=90)
plt.show()


# In[24]:


top_rice_pro_dis = rice_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_rice_pro_dis[:5]
sum_max = top_rice_pro_dis["Production"].sum()
top_rice_pro_dis["precent_of_pro"] = top_rice_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_rice_pro_dis[:5]


# In[25]:


plt.figure(figsize=(18,12))
sns.barplot("District_Name","Production",data=top_rice_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[26]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=rice_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[27]:


sns.jointplot("Area","Production",data=rice_df,kind="reg")


# # Insights:
# From Data Visualization:
# Rice production is mostly depends on Season, Area, State(place).

# # 2. Coconut

# In[28]:


coc_df = data[data["Crop"]=="Coconut "]
print(coc_df.shape)
coc_df[:3]


# In[29]:


sns.barplot("Season","Production",data=coc_df)


# In[30]:


plt.figure(figsize=(13,10))
sns.barplot("State_Name","Production",data=coc_df)
plt.xticks(rotation=90)
plt.show()


# In[31]:


top_coc_pro_dis = coc_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_coc_pro_dis[:5]
sum_max = top_coc_pro_dis["Production"].sum()
top_coc_pro_dis["precent_of_pro"] = top_coc_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_coc_pro_dis[:5]


# In[32]:


plt.figure(figsize=(18,12))
sns.barplot("District_Name","Production",data=top_coc_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[33]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=coc_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[34]:


sns.jointplot("Area","Production",data=coc_df,kind="reg")


# # Insight from Cocunut Production

# * cocunut production is directly proportional to area
# * its production is also gradually increasing over a time of period
# * production is highin kerala state
# * it does not depends on season

# # 3. Sugarcane

# In[35]:


sug_df = data[data["Crop"]=="Sugarcane"]
print(sug_df.shape)
sug_df[:3]


# In[36]:


sns.barplot("Season","Production",data=sug_df)


# In[37]:


plt.figure(figsize=(13,8))
sns.barplot("State_Name","Production",data=sug_df)
plt.xticks(rotation=90)
plt.show()


# In[38]:


top_sug_pro_dis = sug_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_sug_pro_dis[:5]
sum_max = top_sug_pro_dis["Production"].sum()
top_sug_pro_dis["precent_of_pro"] = top_sug_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_sug_pro_dis[:5]


# In[39]:


plt.figure(figsize=(18,8))
sns.barplot("District_Name","Production",data=top_sug_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[40]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=sug_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[41]:


sns.jointplot("Area","Production",data=sug_df,kind="reg")


# # Insighits:
# * Sugarecane production is directly proportional to area
# * And the production is high in some state only.

# # Feature Selection

# In[36]:


data1 = data.drop(["District_Name","Crop_Year"],axis=1)


# In[37]:


data1


# In[38]:


data_dum = pd.get_dummies(data1)
data_dum[:5]


# # Test Train Split

# In[ ]:





# In[39]:


x = data_dum.drop("Production",axis=1)
y = data_dum[["Production"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# In[40]:


x


# In[41]:


y


# In[42]:


x_train[:5]


# # Model -1: Random Forest

# In[43]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)
preds = model.predict(x_test)


# In[44]:


preds


# In[45]:


from sklearn.metrics import r2_score
r = r2_score(y_test,preds)
print("R2score when we predict using Randomn forest is ",r)


# # Model -2 : Linear Regression

# In[46]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[47]:


preds = model.predict(x_test)
preds


# In[48]:


from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test,preds)
r2_score(y_test,preds)


# # Prediction
# Model 1: Randomn forest has high r2score when compare to other model

# In[49]:


tst = test.drop(["District_Name","Crop_Year"],axis=1)
tst_dum = pd.get_dummies(tst)
tst_dum[:5]


# In[50]:


y_test = tst_dum.copy()
print(x_train.shape)
print(y_test.shape)


# In[51]:


def common_member(x_train,x_test): 
    a_set =  set(x_train.columns.tolist())
    b_set =  set(x_test.columns.tolist())
    if (a_set & b_set): 
        return list(a_set & b_set) 


# In[57]:


x_train.columns.tolist()


# In[53]:


x_test.columns.tolist()


# In[54]:


com_fea = common_member(x_train,tst_dum)
len(com_fea)


# In[55]:


com_fea


# In[63]:


y_test.info()


# In[67]:


y_test.columns


# In[68]:


y_test.shape


# In[69]:


y_test


# In[83]:


l=y_test[:1]
l


# In[84]:


p=l[com_fea]
p


# In[98]:


l=l[com_fea]
l


# In[101]:


p['Area']=800
p


# In[102]:


p.columns


# In[91]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train[com_fea],y_train)
preds = model.predict(y_test[com_fea])


# In[92]:


preds


# In[93]:


test["production"] = preds


# In[94]:


test[:10]


# In[95]:


import pickle

# Saving model to disk
pickle.dump(model, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))


# In[58]:


test.to_csv('Prediction.csv')


# In[67]:





# In[5]:


model1


# In[99]:


model1.predict(l)


# In[100]:


model1.predict(p)


# In[ ]:

model = pickle.load(open('/Users/Shreyu/Sonu/Yeild_pred_UI/model1.pkl', 'rb'))
com_fea=['Crop_Jack Fruit', 'Crop_Black pepper', 'State_Name_Karnataka',
       'Season_Rabi', 'Crop_Wheat', 'State_Name_Chhattisgarh',
       'Crop_Groundnut', 'Season_Autumn     ', 'State_Name_Odisha',
       'State_Name_Tamil Nadu', 'State_Name_West Bengal', 'Crop_Potato',
       'Crop_Cotton(lint)', 'Crop_Other Kharif pulses', 'Crop_Safflower',
       'Area', 'State_Name_Nagaland', 'Crop_Arhar/Tur',
       'State_Name_Uttarakhand', 'Crop_Linseed', 'Crop_Maize',
       'State_Name_Chandigarh', 'State_Name_Mizoram', 'Crop_Onion',
       'Crop_Cardamom', 'Crop_Dry chillies', 'Crop_Horse-gram',
       'State_Name_Andhra Pradesh', 'State_Name_Manipur', 'Crop_Bajra',
       'State_Name_Uttar Pradesh', 'Crop_Soyabean', 'Season_Winter     ',
       'Crop_other oilseeds', 'Crop_Peas & beans (Pulses)',
       'State_Name_Haryana', 'Crop_Rice', 'Crop_Niger seed', 'Crop_Banana',
       'Crop_Sesamum', 'Crop_Jute', 'Crop_Cabbage', 'Crop_Moong(Green Gram)',
       'State_Name_Puducherry', 'State_Name_Himachal Pradesh', 'Crop_Mesta',
       'State_Name_Gujarat', 'State_Name_Madhya Pradesh',
       'Crop_Rapeseed &Mustard', 'Crop_Garlic', 'State_Name_Telangana ',
       'Crop_Dry ginger', 'Crop_Blackgram', 'Crop_Cashewnut',
       'Season_Whole Year ', 'State_Name_Andaman and Nicobar Islands',
       'Season_Summer     ', 'State_Name_Goa', 'State_Name_Arunachal Pradesh',
       'Crop_Coconut ', 'Crop_Masoor', 'Crop_Castor seed',
       'State_Name_Rajasthan', 'Crop_Urad', 'State_Name_Maharashtra',
       'State_Name_Jammu and Kashmir ', 'Crop_Pump Kin', 'Crop_Sunflower',
       'Crop_Ragi', 'Crop_Coriander', 'State_Name_Bihar', 'Crop_Guar seed',
       'Crop_Other  Rabi pulses', 'Crop_Small millets', 'Crop_Khesari',
       'Crop_Arecanut', 'Crop_Other Cereals & Millets', 'State_Name_Kerala',
       'Crop_Cowpea(Lobia)', 'Crop_Jowar', 'Season_Kharif     ',
       'Crop_Sugarcane', 'Crop_Turmeric', 'Crop_Gram', 'State_Name_Punjab',
       'Crop_Barley', 'Crop_Tapioca', 'State_Name_Assam', 'Crop_Tobacco',
       'Crop_Sannhamp', 'Crop_Moth', 'Crop_Sweet potato']


dataf=pd.DataFrame(columns=com_fea)
dataf.loc[len(dataf)]=0
    
    
features=['Andhra Pradesh','Kharif','Small millets','1000.00']
dataf['Area']=float(features[3])
    
for j in com_fea:
        test_list=j.strip().split('_')
        
        if (test_list[0]=='State'):
            if(test_list[-1]==features[0]):
                dataf[j]=1
                
        elif (test_list[0]=='Season'):
            if(test_list[-1]==features[1]):
                dataf[j]=1
                
        elif (test_list[0]=='Crop'):            
            if(test_list[-1]==features[2]):
                dataf[j]=1
                    
    
prediction = model.predict(dataf)
print(prediction)
output = round(prediction[0], 2)
output

