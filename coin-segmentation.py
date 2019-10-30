#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import skimage


# In[3]:


import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[4]:


from skimage import data


# In[5]:


coins=data.coins()


# In[6]:


coins


# In[7]:


plt.imshow(coins,cmap='gray')


# In[8]:


from skimage import filters
coins_denoised=filters.median(coins,selem=np.ones((5,5)))
f,(ax0,ax1)=plt.subplots(1,2,figsize=(15,5))
ax0.imshow(coins)
ax1.imshow(coins_denoised)


# In[9]:


from skimage import feature
edges=skimage.feature.canny(coins,sigma=3)
plt.imshow(edges)


# In[10]:


from scipy.ndimage import distance_transform_edt
dt=distance_transform_edt(~edges)
plt.imshow(dt)


# In[11]:


local_max=feature.peak_local_max(dt,indices=False,min_distance=5)
plt.imshow(local_max,cmap='gray')


# In[12]:


peak_idx=feature.peak_local_max(dt,indices=True,min_distance=5)
peak_idx[:5]


# In[13]:


plt.plot(peak_idx[:,1],peak_idx[:,0],'r.')
plt.imshow(dt)


# In[14]:


from skimage import measure
markers=measure.label(local_max)


# In[15]:


markers


# In[16]:


from skimage import morphology,segmentation
labels=morphology.watershed(-dt,markers)
plt.imshow(segmentation.mark_boundaries(coins,labels))


# In[17]:


from skimage import color
plt.imshow(color.label2rgb(labels,image=coins))


# In[18]:


plt.imshow(color.label2rgb(labels,image=coins,kind='avg'),cmap='gray')


# In[19]:


regions=measure.regionprops(labels,intensity_image=coins)


# In[20]:


regions


# In[21]:


region_means=[r.mean_intensity for r in regions]
plt.hist(region_means,bins=20)


# In[22]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
region_means=np.array(region_means).reshape(-1,1)


# In[23]:


model.fit(region_means)
print(model.cluster_centers_)


# In[24]:


bg_fg_labels=model.predict(region_means)
bg_fg_labels


# In[25]:


classified_labels=labels.copy()
for bg_fg,region in zip(bg_fg_labels,regions):
    classified_labels[tuple(region.coords.T)]=bg_fg


# In[26]:


plt.imshow(color.label2rgb(classified_labels,image=coins))


# In[27]:


plt.imshow(coins,cmap='gray')


# In[28]:


from skimage import filters
coins_denoised=filters.median(coins,selem=np.ones((5,5)))
f,(ax0,ax1)=plt.subplots(1,2,figsize=(15,5))
ax0.imshow(coins)
ax1.imshow(coins_denoised)


# In[29]:


from skimage import feature
edges=skimage.feature.canny(coins,sigma=3)
plt.imshow(edges)


# In[30]:


from scipy.ndimage import distance_transform_edt
dt=distance_transform_edt(~edges)
plt.imshow(dt)


# In[31]:


local_max=feature.peak_local_max(dt,indices=False,min_distance=5)
plt.imshow(local_max,cmap='gray')


# In[32]:


peak_idx=feature.peak_local_max(dt,indices=True,min_distance=5)
peak_idx[:5]


# In[33]:


plt.plot(peak_idx[:,1],peak_idx[:,0],'r.')
plt.imshow(dt)


# In[34]:


from skimage import measure
markers=measure.label(local_max)


# In[35]:


from skimage import morphology,segmentation
labels=morphology.watershed(-dt,markers)
plt.imshow(segmentation.mark_boundaries(coins,labels))


# In[36]:


from skimage import color
plt.imshow(color.label2rgb(labels,image=coins))


# In[37]:


plt.imshow(color.label2rgb(labels,image=coins,kind='avg'),cmap='gray')


# In[38]:


regions=measure.regionprops(labels,intensity_image=coins)


# In[39]:


region_means=[r.mean_intensity for r in regions]
plt.hist(region_means,bins=20)


# In[40]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
region_means=np.array(region_means).reshape(-1,1)


# In[41]:


model.fit(region_means)
print(model.cluster_centers_)


# In[42]:


bg_fg_labels=model.predict(region_means)
bg_fg_labels


# In[43]:


classified_labels=labels.copy()
for bg_fg,region in zip(bg_fg_labels,regions):
    classified_labels[tuple(region.coords.T)]=bg_fg


# In[44]:


plt.imshow(color.label2rgb(classified_labels,image=coins))


# In[ ]:





# In[ ]:




