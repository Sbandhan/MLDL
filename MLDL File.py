#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


from mlxtend.data  import  loadlocal_mnist
import platform
import numpy as np 
import pandas as pd
import matplotlib.pyplot as  plt 
import  seaborn as  sns
import numpy as np


# In[104]:


import gzip


# In[146]:


## 2 PCA
X,y = loadlocal_mnist(images_path="train-images.idx3-ubyte",labels_path="train-labels.idx1-ubyte")


# In[147]:


np.savetxt(fname='images.csv', X=X,  delimiter=',',  fmt='%d')
np.savetxt(fname='labels.csv',X=y,  delimiter=',',  fmt='%d')


# In[148]:


df_img =  pd.read_csv('images.csv')
df_img.head()


# In[149]:


df_label = pd.read_csv('labels.csv')
df_label.rename(columns={'5':  'label'},inplace = True)
df_label.head(10)


# In[150]:


label = df_label["label"]


# In[151]:


ind= np.random.randint(0,20000)
plt.figure(figsize=(20,5))
grid_data =  np.array(df_img.iloc[ind]).reshape(28,28)
plt.imshow(grid_data,  interpolation = None,  cmap =  'gray') 
plt.show()
print(label[ind])


# In[152]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
std_df=  scaler.fit_transform(df_img)
std_df.shape


# In[153]:


covar_mat=  np.matmul(std_df.T,  std_df)
covar_mat.shape


# In[154]:


from scipy.linalg import eigh
values,  vectors= eigh(covar_mat,  eigvals=(782,783))
print("Dimensions of  Eigen vector:",  vectors.shape)
vectors= vectors.T
print("Dimensions of Eigen vector:",  vectors.shape)


# In[155]:


final_df= np.matmul(vectors,  std_df.T)
print("vectros:",  vectors.shape,  "n",  "std_df:",  std_df.T.shape,  "n",  "final_df:",final_dfT.T.shape)


# In[156]:


final_dfT=  np.vstack((final_df,  label)).T
dataFrame=  pd.DataFrame(final_dfT,  columns=['pca_1',   'pca_2',   'label'])
dataFrame


# In[178]:


sns.FacetGrid(dataFrame,  hue=  'label',).map(sns.scatterplot,   'pca_1','pca_2')
plt.show()


# In[158]:


#problem 2 PCA (A)


# In[159]:


train_df=  pd.read_csv("train.csv")
train_df.head()


# In[160]:


a=train_df['label']
b=train_df.drop(columns='label')


# In[161]:


mean =  train_df.groupby(['label']).mean()
print((mean))


# In[162]:


std =  train_df.groupby(['label']).std()
print((std))


# In[163]:


x=  train_df.drop(['label'],  axis=1)# Displaying the  images with mean of each pi
y= train_df['label']
from sklearn.decomposition  import  PCA
pca =  PCA(n_components=2)# Dimensions 10
principalCom=  pca.fit_transform(x)
principalDf=  pd.DataFrame(data=  principalCom, columns=['pca_1',   'pca_2'])
df2=  pd.concat([principalDf,  train_df[['label']]],  axis=1)


# In[164]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
c_map= plt.cm.get_cmap('jet',10)
plt.scatter(principalCom[:,0],  principalCom[:,1],  s=15,cmap= c_map, c= df2['label'])
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2') 
plt.show()


# In[165]:


import numpy as np
import matplotlib.pyplot as  plt


def image2vec(image):
    return  image.flatten()


def vec2image(vec):
    return vec.reshape(28, 28)

images1=b
labels1=a

def read_mnist(normalize=True):
    labels=labels1
    images=images1
    #labels = np.load('t10k-labels.idx1-ubyte',allow_pickle=True).astype(int) 
    #images =  np.load('t10k-images.idx3-ubyte',allow_pickle=True).astype(float) 
    print(labels)
    print(images)
    if normalize:
        images=  images/255# rescale to be between 0 and 1
    return  images,  labels


class MnistPlotter:
    

# plot params  to make figures nice
    size=28
    cmap='Greys'
    dpi=96
    lw= dpi/(1024*32)

def draw_image(self,  image,  label=None):
# params
    figsize=(6,6)

# pLot
    fig = plt.figure(figsize=figsize,  dpi=self.dpi) 
    ax1 = fig.add_subplot(111)
    self._draw_single_image(ax1,  image,  label)

    return fig
def draw_two_images(self,  im1,  im2): 
# params
    figsize=(12,6)

# plot
    fig= plt.figure(figsize=figsize,  dpi=self.dpi)
    ax1= fig.add_subplot(121)
    ax2= fig.add_subplot(122)
    for ax,  image in  zip([ax1,  ax2],[im1,  im2]):
        self._draw_single_image(ax,  image)

    return fig

def draw_three_images(self,  im1,  im2,  im3, **kwargs):

# params
 


    figsize=(18,6)

# pLot
    fig= plt.figure(figsize=figsize,  dpi=self.dpi)
    ax1= fig.add_subplot(131)
    ax2= fig.add_subplot(132)
    ax3= fig.add_subplot(133)

    for ax,  image in  zip([ax1,  ax2,  ax3],[im1,  im2,  im3]):
        self._draw_single_image(ax,  image,**kwargs)

    return fig

def _draw_single_image(self,  ax,  image,  label=None,  fs=50, **kwargs):

# defauLt kwargs
    default_kwargs = dict(vmin=0,  vmax=1,  edgecolor='k',  lw=self.lw) 
    default_kwargs.update(kwargs)

# params
    lw_border=6
    lim=[0,  self.size]

# pLot image
    cmap= plt.get_cmap(self.cmap)
    ax.pcolormesh(image,  cmap=cmap,**default_kwargs)

# draw boundaries
    for v in [0,  self.size]:
        ax.plot([v,  v],  lim,  lw=lw_border,  color='k') 
        ax.plot(lim, [v,  v],  lw=lw_border,  color='k')

# handLe axis
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_yaxis()
    ax.set_aspect('equal') 
    ax.set_axis_off()

# draw LabeL
#if label is not None:
#ax.text(0.03, 0.84,  ctr(label),  fontsize=fs,  color='k',  transform=ax.


# In[166]:


from  sklearn.metrics  import  r2_score
from  sklearn.metrics  import  mean_squared_error 
from  math  import  sqrt
import numpy as np

#where X is the originaL  data and f is  the compressed data.
X=  dataFrame.head(59999)
f=  final_dfT
r2=  r2_score(X,f)
rmse=  sqrt(mean_squared_error(X,f))


# In[167]:


pca =  PCA()
pca.fit(x)


# In[168]:


pca.n_components_


# In[169]:


tot=  sum(pca.explained_variance_)
tot


# In[170]:


var_exp = [(i/tot)*100 for i in  sorted(pca.explained_variance_,  reverse=True)]
print(var_exp[0:5])


# In[171]:


tot=  sum(pca.explained_variance_)
tot


# In[172]:


var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_,  reverse=True)]
print(var_exp[0:5])


# In[173]:


cum_var_exp =  np.cumsum(var_exp)


# In[174]:


plt.figure(figsize=(10, 5))
plt.step(range(1, 785),  cum_var_exp,  where='mid',label='cumulative explained var')
plt.title('Cumulative Explained Variance as  a  Function of the Number of Component')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axhline(y=95,  color='k',  linestyle='--',  label=  '95% Explained Variance')
plt.axhline(y=90,  color='c',  linestyle='--',  label=  '90% Explained Variance')
plt.axhline(y=85,  color='r',  linestyle='--',  label=  '85% Explained Variance')
plt.legend(loc='best') 
plt.show()


# In[ ]:




