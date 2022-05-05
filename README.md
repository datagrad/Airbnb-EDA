# Airbnb-EDA



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the Data from csv

data = pd.read_csv('https://raw.githubusercontent.com/datagrad/Airbnb-EDA/main/AB_NYC_2019.csv')

print('Number of features: %s' %data.shape[1])
print('Number of examples: %s' %data.shape[0])


# Display Head and Tail Data

data.head().append(data.tail())


# Checking Null and Data Type using info()

data.info()


# Describe for Numerical Distribution



data.describe()


# Analysis-Top_3_hosts

top_3_hosts = (pd.DataFrame(data.host_id.value_counts())).head(3)
top_3_hosts.columns=['Listings']
top_3_hosts['host_id'] = top_3_hosts.index
top_3_hosts.reset_index(drop=True, inplace=True)
top_3_hosts

# Analysis-Top_3_neighbourhoood_groups

top_3_neigh = pd.DataFrame(data['neighbourhood_group'].value_counts().head(3))
top_3_neigh.columns=['Listings']
top_3_neigh['Neighbourhood Group'] = top_3_neigh.index
top_3_neigh.reset_index(drop=True, inplace=True)
top_3_neigh

# WordCloud

from wordcloud import WordCloud, ImageColorGenerator
wordcloud = WordCloud(
                          background_color='white'
                         ).generate(" ".join(data.neighbourhood))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()

# Drop Columns

data.drop(['id','host_id','host_name','last_review'],axis=1,inplace=True)


# NULL Count in each column

data.isnull().sum()


# Data Distribution Check

data_check_distrib=data.drop(data[pd.isnull(data.reviews_per_month)].index)

{"Mean":np.nanmean(data.reviews_per_month),"Median":np.nanmedian(data.reviews_per_month),
 "Standard Dev":np.nanstd(data.reviews_per_month)}


# Imputation with Median

def impute_median(series):
    return series.fillna(series.median())


# Histogram 

plt.hist(data_check_distrib.reviews_per_month,  bins=50)
plt.title("Distribution of reviews_per_month")
plt.xlim((min(data_check_distrib.reviews_per_month), max(data_check_distrib.reviews_per_month)))



def impute_median(series):
    return series.fillna(series.median())

data.reviews_per_month=data["reviews_per_month"].transform(impute_median)


# Correlation Matrix Plot


data['reviews_per_month'].fillna(value=0, inplace=True)

f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax, cmap='Reds')
plt.show()


# Histogram

fig = plt.figure(figsize = (15,10))
ax = fig.gca()
data.hist(ax=ax)
plt.show()


# Pie Plot

labels = data.neighbourhood_group.value_counts().index
colors = ['lightblue','beige','lightgreen','orange','cyan']
explode = [0,0,0,0,0]
sizes = data.neighbourhood_group.value_counts().values

plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=[0.1,0.0,0.3,0.5,0.0], labels=labels, colors=colors, autopct='%1.1f%%',shadow=True)
plt.title('Neighbourhood Group',color = 'black',fontsize = 15)
plt.show()


# Barplot neighbourhood_group-price
result = data.groupby(["neighbourhood_group"])['price'].aggregate(np.median).reset_index().sort_values('price')
sns.barplot(x='neighbourhood_group', y="price", data=data,palette=colors, order=result['neighbourhood_group']) 
plt.xticks(rotation=45)
plt.show()


# Boxplot neighbourhood_group-availability_365
result = data.groupby(["neighbourhood_group"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')
sns.boxplot(x='neighbourhood_group', y="availability_365", data=data) 
plt.show()


# Pie Plot Room-Type

labels = data.room_type.value_counts().index
colors = ['lightblue','pink','beige']
explode = [0,0,0]
sizes = data.room_type.value_counts().values


plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=[0,0.05,0.5], labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
# plot.pie(explode=,autopct='%1.1f%%',ax=ax[0],)
plt.title('Room-Type',color = 'Brown',fontsize = 15)
plt.show()


# Bar Plot room_type-price
result = data.groupby(["room_type"])['price'].aggregate(np.median).reset_index().sort_values('price')
sns.barplot(x='room_type', y="price", data=data, order=result['room_type']) 
plt.show()


# Box Plot room_type-availability_365
result = data.groupby(["room_type"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')
sns.boxplot(x='room_type', y="availability_365", data=data, order=result['room_type']) 
plt.show()


# Line Plot Availability vs Price 
sns.lineplot(x='availability_365',y='price',data=data)
plt.show()


# Scatter Plot Neighbourhood Group

plt.figure(figsize=(10,6))
sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group)
plt.ioff()


# Scatter Plot

plt.figure(figsize=(10,6))
sns.scatterplot(data.longitude,data.latitude,hue=data.availability_365)
plt.ioff()
