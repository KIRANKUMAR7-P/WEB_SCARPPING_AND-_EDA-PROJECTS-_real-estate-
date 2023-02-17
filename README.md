# WEB_SCARPPING_AND-_EDA-PROJECTS-_real-estate-
<div style="background-color:#00FFFF;padding:14px;">
    <p style="text-align:center"><font size="12px" color="#800080" face="Product Sans"><b>Exploring the Hyderabad Real Estate Market: A Web Scraping and Data Analysis Project.</b></font> </p>
</div>

## Problem Statement:

To analyze the real estate market in Hyderabad, a web scraping and exploratory data analysis project will be conducted using the website magicbricks.com. 
The goal of this project is to gather and analyze data on new real estate projects in Hyderabad, including information on location, price range, amenities, and developer. 
The collected data will be used to identify trends and patterns in the real estate market, and to understand the current state of the market in Hyderabad.
Additionally, the project will aim to determine the most desirable areas and features for new real estate projects in Hyderabad, and to identify potential opportunities for investors and developers.

<div style="background-color:#FFB6C1;padding:14px">
<p style="text-align:center"><font size="12px" color="#006400" face="Product Sans"><b>Web Scrapping</b></font> </p>
</div>

import pandas as pd 
import numpy as np
import requests
from bs4 import BeautifulSoup

# Identify the URL
URL='https://www.magicbricks.com/new-projects-Hyderabad'

# # Loading the WebPage in Memory using requests library

# requests.get(URL) 
page=requests.get(URL)

# checking the status of url page
page.status_code

## Extracting the HTML Code of the WebPage | text
htmlcode=page.text

htmlcode

# Format the HTML code using bs4 library
soup=BeautifulSoup(htmlcode)

# prettify method give you readable html code
print(soup.prettify)

## Taking tag and class from the website 

#### company --> tag = 'p'  class = "proHeading"
 
#### price ---> tag = "div"  class = "proPriceField"
    
#### location ---> tag = "p"  class = "proGroup"
    
#### published_by---> tag = "p" class = "proGroup projDeveloper"
    
#### bedroom---> tag = "div" class = "proDescColm1"
    
#### Floors---> tag = "div" class = "floorPlans"

#### Construction_status---> tag = "div" class = "proDetailsRowElm posByCons"

#### Area---> tag = "div" class = "proDescColm2"

# find

company=soup.find('p',attrs={"class":"proHeading"})

company.text

price=soup.find("div",attrs={"class":"proPriceField"})

price.text



# Find_all

soup.find_all("p",attrs={"class":"proGroup"})

soup.find_all("div",attrs={"class":"proDescColm1"})

soup.find_all("div",attrs={"class":"floorPlans"})

### Generating pages for the URL 

# Generating the url to all the required pages.
"""URL = 'https://www.magicbricks.com/new-projects-Hyderabad'"""
for i in range(1,45):
    print("https://www.magicbricks.com/new-projects-Hyderabad/page-{}".format(i))

# Code for Web Scrapping (Incorrect way)¶

# Taking all the information from the website and appending into the below mentioned lsts.
company = []
price = []
location = []
published = []
floor = []
bedroom = []
construction = []
Area = []


for i in range(1,80):
    URL = "https://www.magicbricks.com/new-projects-Hyderabad/page-{}".format(i) 
    page = requests.get(URL)
    htmlCode = page.text
    
    soup = BeautifulSoup(htmlCode)
    
    # title
    companies = soup.find_all("p",attrs={"class":"proHeading"})
    for item in companies:
        company.append(item.text)
        
   
        
    # prices
    prices = soup.find_all("div",attrs={"class":"proPriceField"})
    for item in prices:
        price.append(item.text)
        
    # features
    locations = soup.find_all("p",attrs={"class":"proGroup"})
    for item in locations:
        location.append(item.text)
        
    published_by = soup.find_all("p",attrs={"class":"proGroup projDeveloper"})
    for item in published_by:
        published.append(item.text)
        
    floors = soup.find_all("div",attrs={"class":"floorPlans"})
    for item in floors:
        floor.append(item.text)
        
    bedrooms = soup.find_all("div",attrs={"class":"proDescColm1"})
    for item in bedrooms:
        bedroom.append(item.text)
        
    construction_status = soup.find_all("div",attrs={"class":"proDetailsRowElm posByCons"})
    for item in construction_status:                                               
        construction.append(item.text)
        
    Area_sq_ft = soup.find_all("div",attrs={"class":"proDescColm2"})
    for item in Area_sq_ft:
        Area.append(item.text)
        
    

print(len(company))
print(len(price)) 
print(len(location))
print(len(published))
print(len(floor)) 
print(len(bedroom))
print(len(construction))
print(len(Area))

# Code for Web Scrapping (Correct way)¶

company = []
price = []
location = []
published = []
floor = []
bedroom = []
construction = []
Area = []


for i in range(1,80):
    URL = "https://www.magicbricks.com/new-projects-Hyderabad/page-{}".format(i) 
    page = requests.get(URL)
    htmlCode = page.text
    
    soup = BeautifulSoup(htmlCode)
    
    for x in soup.find_all("div", attrs={"class":"srpBlockListRow"}):
    
        # title
        companies = x.find("p",attrs={"class":"proHeading"})
        if companies is None:
            company.append(np.NaN)
        else:
            company.append(companies.text)
        
       
        # prices
        prices = x.find("div",attrs={"class":"proPriceField"})
        if prices is None:
            price.append(np.NaN)
        else:
            price.append(prices.text)
        
        # features
        locations = x.find("p",attrs={"class":"proGroup"})
        if locations is None:
            location.append(np.NaN)
        else:
            location.append(locations.text)
        
        published_by = x.find("p",attrs={"class":"proGroup projDeveloper"})
        if published_by is None:
            published.append(np.NaN)
        else:
            published.append(published_by.text)
        
        floors = x.find("div",attrs={"class":"floorPlans"})
        if floors is None:
            floor.append(np.NaN)
        else:
            floor.append(floors.text)
        
        bedrooms = x.find("div",attrs={"class":"proDescColm1"})
        if bedrooms is None:
            bedroom.append(np.NaN)
        else:
            bedroom.append(bedrooms.text)
        
        construction_status = x.find("div",attrs={"class":"proDetailsRowElm posByCons"})
        if construction_status is None:
            construction.append(np.NaN)
        else:
            construction.append(construction_status.text)
        
        Area_sq_ft = x.find("div",attrs={"class":"proDescColm2"})
        if Area_sq_ft is None:
            Area.append(np.NaN)                          
        else:
            Area.append(Area_sq_ft.text)
            
            
    

print(len(company))
print(len(price)) 
print(len(location))
print(len(published))
print(len(floor)) 
print(len(bedroom))
print(len(construction))
print(len(Area))



# Creating a DataFrame and saving it in a csv file

df = pd.DataFrame({"companies":company, "prices":price,"locations":location, "published_by":published, "floors":floor, "bedrooms":bedroom, "Possession":construction, "Area_sq_ft":Area})

df.to_csv('Real_state_Webscrapping_and_EDA_project2.csv', index = False)

<div style="background-color:#224256;padding:14px">
<p style="text-align:center"><font size="12px" color="#ed8e0f" face="Product Sans"><b>Exploratory Data_Analysis</b></font> </p>
</div>

<div style="background-color:#97e2b6;padding:14px">
<p style="text-align:center"><font size="12px" color="blue" face="Product Sans"><b>DataFrame Cleaning</b></font> </p>
</div>



# Extracting the Details using Regex

import numpy as np
import pandas as pd

import re

df = pd.read_csv("Real_state_Webscrapping_and_EDA_project2.csv")

df.head()

import pandas as pd

df = pd.read_csv("Real_state_Webscrapping_and_EDA_project2.csv")
df.dropna(subset=["prices"], axis=0, how='any', inplace=True)
df.to_csv("Real_state_Webscrapping_and_EDA_project2.csv", index=False)


df.shape

df.info()



# replacing '\n' from the columns with ''

# replacing '\n' with '' and assigning in to column
df['companies']=df['companies'].apply(lambda x: x.replace('\n','').replace('\t', ''))

df['companies']

df['locations']=df['locations'].str.replace('\W+[?:n|t]*', ' ', regex = True)

df['locations'].head()

df['published_by']=df['published_by'].str.replace('\n','').str.replace('\t', '')

df['published_by'].head()

df['bedrooms'] = df['bedrooms'].str.replace('\W+[?:n|t]*', ' ', regex = True)

df['bedrooms'].head()

df['Possession']=df['Possession'].apply(lambda x: x.replace('\n','').replace('\t',''))

df['Possession'].head()

df['Possession']=df['Possession'].str.replace("Possession in","")
df['Possession']

df['Area_sq_ft'] = df['Area_sq_ft'].str.replace('\W+[?:n|t]*', ' ', regex = True)
df['Area_sq_ft'].head()


df['Area_sq_ft']= df['Area_sq_ft'].str.replace('\s[a-zA-Z]+', ' ', regex = True)
df['Area_sq_ft']

df['start_Area_sq_ft']=df['Area_sq_ft'].str.split().str[0]
df['start_Area_sq_ft']

df['final_Area_sq_ft']=df['Area_sq_ft'].str.split().str[1]
df['final_Area_sq_ft']

df['prices']



# Regular expression

import re

reg = r'^[\d]+[\s.\d]+[a-zA-z]+'
df['start_price']=df['prices'].apply(lambda x: re.findall(reg,str(x)))

df['start_price']

reg= r'[-][\s]+[\d.\s]+[a-zA-Z]+'
df['final_price']=df['prices'].apply(lambda x: re.findall(reg,str(x)))

df['final_price']

df['locations']

df['locations']=df['locations'].str.replace('in', '' )
df['locations']



df['locality']=df['locations'].str.split().str[0]
df['locality']

df['locations']=df['locations'].str.extract(r'(\bHyderabad\b)')
df['locations']

regex=r'^[\d]+'
df['number_floors']=df['floors'].apply(lambda x: re.findall(regex,str(x)))

df['number_floors']

df['number_floors'].isnull().value_counts()

df['bedrooms'].isnull().value_counts()

df['locations'].isnull().value_counts()

df


<div style="background-color:lightgreen;padding:14px">
<p style="text-align:center"><font size="12px" color="red" face="Product Sans"><b>Data manupulation</b></font> </p>
</div>

df['start_price'] = df['start_price'].apply(lambda x : ''.join(x))

df['start_price']=df['start_price'].apply(lambda x: float(x.replace('Lac',''))*100000 if 'Lac' in x else float(x.replace('Cr',''))*10000000 if "Cr" in x else float(x.replace('','0')))

df['start_price']

df['final_price']

df['final_price'] = df['final_price'].apply(lambda x : ''.join(x))

df['final_price']=df['final_price'].fillna(np.nan).astype(str).apply(lambda x: float(x.replace('Lac','').replace('-',''))*100000 if 'Lac' in x else float(x.replace('Cr','').replace('-',''))*10000000 if 'Cr' in x else x.replace('','0'))

df['final_price']=df['final_price'].astype(object).astype(float)

df['final_price']

df['locality'] = df['locality'].apply(lambda x : ''.join(x))

df['locality']

import re

df['bedrooms'] = df['bedrooms'].astype(str)
df['bedrooms'] = df['bedrooms'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if x is not None and re.search(r'\d+', str(x)) is not None else np.NaN)
df['bedrooms'].fillna(value=df['bedrooms'].mean(), inplace=True)
df['bedrooms'] = df['bedrooms'].astype('int64')



df['bedrooms']


df['number_floors'] = df['number_floors'].apply(lambda x : ''.join(x))

df['number_floors']

df['published_by']=df['published_by'].str.replace("by","")

df['published_by']

df

df["published_by"].fillna(value='Unknown', inplace=True)

df["published_by"]



df.info()

df['number_floors']=df['number_floors'].replace('',np.nan)
df['number_floors']

df['number_floors']=df['number_floors'].fillna(0).astype(object).astype('int64')

df['number_floors']

df['number_floors'].isnull().value_counts()

df['bedrooms'].isnull().value_counts()

df['start_price'].isnull().value_counts()



df['final_price'].isnull().value_counts()

# Droping the columns

df.drop('prices', inplace=True, axis=1)
df.drop('floors', inplace=True, axis=1)
#df.drop('bedrooms',inplace=True, axis=1)

df.info()

df

df["bedrooms"]=df["bedrooms"].astype(float)
mean =df["bedrooms"].mean()
df["bedrooms"].fillna(value=mean, inplace=True)
df["bedrooms"] = df["bedrooms"].astype(int)
df["bedrooms"]


df["final_Area_sq_ft"] = df["final_Area_sq_ft"].astype(float)
mean = df["final_Area_sq_ft"].mean()
df["final_Area_sq_ft"].fillna(value=mean, inplace=True)
df["final_Area_sq_ft"] = df["final_Area_sq_ft"].astype(int)
df["final_Area_sq_ft"]


df["start_Area_sq_ft"] = df["start_Area_sq_ft"].astype(float)
mean = df["start_Area_sq_ft"].mean()
df["start_Area_sq_ft"].fillna(value=mean, inplace=True)
df["start_Area_sq_ft"] = df["start_Area_sq_ft"].astype(int)
df["start_Area_sq_ft"]

df.drop('Area_sq_ft', inplace=True, axis=1)

df.describe()

df.info()

<div style="background-color:olive;padding:14px">
<p style="text-align:center"><font size="12px" color="#d11f2f" face="Product Sans"><b>Data Visiualization</b></font> </p>
</div>

import matplotlib.pyplot as plt
import seaborn as sns

df

#  Uni-variate Analysis:


Univariate analysis is a statistical method that focuses on the analysis of a single variable at a time. It is a basic form of exploratory data analysis that involves examining the distribution, central tendency, and dispersion of a single variable. Univariate analysis is useful in identifying patterns, relationships, and outliers in a dataset, and is often used as a preliminary step before conducting more complex multivariate analysis.

# single numerical


import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact

@interact(col1=df.select_dtypes('number').columns)
def groupby_(col1):
    sns.displot(data=df[col1], kind='hist', bins=40, color='purple')
    plt.legend(title=f"Distribution of {col1}")
    plt.show()

bedroom_counts = df['bedrooms'].value_counts()
labels = ['2 bhk/plot', '3 bhk/plot', '4 bhk/plot', '5 bhk/plot', '1 bhk/plot']
colors = ['#D7BDE2', '#A569BD', '#7D3C98', '#4A235A', '#1F618D']
explode = [0.1, 0.1, 0.1, 0.2, 0.8]
plt.pie(bedroom_counts, labels=labels, colors=colors, explode=explode, autopct='%0.2f%%', startangle=90,radius=0.7, pctdistance=0.7)
plt.axis('equal')
plt.legend(title="Number of Bedrooms", loc="upper left", fontsize=10)
plt.show()


The majority of the bedrooms of bhk/plot, as indicated by the pie chart, are 2 bhk/plot with the highest percentage occupancy.

# single categorical 


@interact(col1=df.select_dtypes('object').columns)
def groupby_(col1):
    plt.figure(figsize=(25,10))
    sns.countplot(x=df[col1])
    plt.legend(title=f"Distribution of {col1}")
    plt.xticks(rotation=90)
    plt.show()

According to the countplot, the majority of the possessions are 'Ready To Move.'

plt.figure(figsize=(25,10))
plt.title("Locality Distribution of Properties in Hyderabad Real Estate Market",fontsize=40)
a=sns.countplot(x=df['locality'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.show()

The countplot indicates that the majority of the constructions took place in the area of 'KOMPALLY.'

# Bi-variate Analysis and Multivariate :


Bi-variate Analysis: A statistical method that examines the relationship between two variables.

Multivariate Analysis: A statistical method that examines the relationship between three or more variables. It provides a deeper understanding of the relationships between variables and can uncover complex patterns and interactions between variables.

# numerical to numerical


@interact(col1=df.select_dtypes('number').columns,
         col2=df.select_dtypes('number').columns)
def groupby_(col1,col2):
    sns.barplot(data=df, y=df[col1],x=df[col2])
    plt.legend(title=f"Relationship between {col1}and {col2}")
    plt.show()




plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.title("Relationship between bedrooms and start_price")
a=sns.barplot(data=df, y=df['start_price'],x=df['bedrooms'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(8,6))
plt.subplot(1,2,2)
plt.title("Relationship between bedrooms and final_price")
a=sns.barplot(data=df, y=df['final_price'],x=df['bedrooms'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

The barplot displays that the average price for 1 bhk/plot is comparatively low while the average price for 5 bhk/plots is the highest.

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.title("Relationship between bedrooms and start_Area_sq_ft")

a=sns.barplot(data=df, y=df['start_Area_sq_ft'],x=df['bedrooms'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(8,6))
plt.subplot(1,2,2)
plt.title("Relationship between bedrooms and final_Area_sq_ft")
a=sns.barplot(data=df, y=df['final_Area_sq_ft'],x=df['bedrooms'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

The barplot demonstrates that the start area in square feet for 5 bhk/plots is relatively high compared to that of 1 bhk/plots, which is low.

The barplot demonstrates that the final area in square feet for 2 bhk/plots is relatively high compared to that of 3 bhk/plots, which is low.

# categorical to numerical


# Largest 100 Properties by Start Area (Square Feet)

df1=df.nlargest(100,'start_Area_sq_ft')
df1



#Smallest 100 Properties by Start Area (Square Feet)

df2=df.nsmallest(100,"start_Area_sq_ft")
df2

df3=df.nsmallest(100,"final_Area_sq_ft")
df3

df4=df.nlargest(100,"final_Area_sq_ft")
df4

plt.figure(figsize=(50,8))
plt.subplot(1,2,1)
plt.title("Relationship between bedrooms and locality",fontsize=40)
a=sns.barplot(data=df, x=df['locality'],y=df['bedrooms'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(50,8))
plt.subplot(1,2,2)
plt.title("Relationship between locality and number_floors",fontsize=40)
a=sns.barplot(data=df, x=df['locality'],y=df['number_floors'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()



In Kandukur, the majority of the bedrooms are occupied by 5 BHK/plot. In Naknumpur, the majority of the bedrooms are occupied by 1 BHK/plot. However, in most areas, the majority of the bedrooms are occupied by 2 BHK/plots.

In Nacharam, the construction height is represented by the highest number of floors in the bar plot. In most areas, the construction height is represented by zero floors.

# numerical to numerical 
# largest 100 start_price compared to area of squre feet


plt.figure(figsize=(50,8))
plt.subplot(1,2,1)
plt.title("Relationship between start_price and start_Area_sq_ft",fontsize=40)
a=sns.barplot(data=df1, y=df1['start_Area_sq_ft'],x=df1['start_price'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(50,8))
plt.subplot(1,2,2)
plt.title("Relationship between final_Area_sq_ft and start_price",fontsize=40)
a=sns.barplot(data=df4, y=df4['final_Area_sq_ft'],x=df4['start_price'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

"The largest 100 properties in terms of both starting price and area square feet in the Hyderabad real estate market are those with area square feet above 40,000 sqft and prices above 4,000,000, according to the barplots."



#smallest 100 startprice compare to area of square feet.

plt.figure(figsize=(50,8),dpi=300)
plt.subplot(1,2,1)
plt.title("Relationship between start_price and start_Area_sq_ft",fontsize=40)
a=sns.barplot(data=df2, y=df2['start_Area_sq_ft'],x=df2['start_price'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(50,8),dpi=300)
plt.subplot(1,2,2)
plt.title("Relationship between final_Area_sq_ft and start_price",fontsize=40)
a=sns.barplot(data=df3, y=df3['final_Area_sq_ft'],x=df3['start_price'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

"The smallest 100 properties in terms of both starting price and area square feet in the Hyderabad real estate market are those with area square feet below 100 sqft and prices below 6,80,000, according to the barplots."


# largest 100 final_price compared to area of squre feet

plt.figure(figsize=(50,8))
plt.subplot(1,2,1)
plt.title("Relationship between final_price and start_Area_sq_ft",fontsize=30)
a=sns.barplot(data=df1, y=df1['start_Area_sq_ft'],x=df1['final_price'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(50,8))
plt.subplot(1,2,2)
plt.title("Relationship between final_Area_sq_ft and final_price",fontsize=30)
a=sns.barplot(data=df4, y=df4['final_Area_sq_ft'],x=df4['final_price'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

"The largest 100 properties in terms of both final price and area square feet in the Hyderabad real estate market are those with area square feet above 40,000 sqft and prices above 24,00,00,000, according to the barplots."


#smallest 100 final price compare to area of square feet.

plt.figure(figsize=(50,8))
plt.subplot(1,2,1)
plt.title("Realatioship between final_price and start_Area_sq_ft",fontsize=40)
a=sns.barplot(data=df2, y=df2['start_Area_sq_ft'],x=df2['final_price'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(50,8))
plt.subplot(1,2,2)
plt.title("Realatioship between final_Area_sq_ft and final_price",fontsize=40)
a=sns.barplot(data=df3, y=df3['final_Area_sq_ft'],x=df3['final_price'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

"The smallest 100 properties in terms of both final price and area square feet in the Hyderabad real estate market are those with low area square feet below 100 sqft and prices below 54,00,000, according to the barplots."


# largest 100 prices compare to companies.
dfps=df.nsmallest(100,"start_price")
print(dfps)
dffs=df.nlargest(100,"start_price")
print(dffs)

dfps1=df.nsmallest(100,"final_price")
print(dfps1)
dffs1=df.nlargest(100,"final_price")
print(dffs1)

plt.figure(figsize=(21,9))
plt.subplot(2,1,1)
plt.title("Relationship between companies and start_price",fontsize=30)
a=sns.barplot(data=dffs, y=dffs['start_price'],x=dffs['companies'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(21,9))
plt.subplot(2,1,2)
plt.title("Relationship between companies and final_price",fontsize=30)
a=sns.barplot(data=dffs1, y=dffs1['final_price'],x=dffs1['companies'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

"The largest 100 properties in terms of starting price, as shown in the bar plot, are associated with the Capital 45 company, which has the highest starting prices in the Hyderabad real estate market."

"The largest 100 properties in terms of final price, as shown in the bar plot, are associated with the Hitech Green Farms company, which has the highest final prices in the Hyderabad real estate market."

# smallest 100 prices compare to companies.

plt.figure(figsize=(21,9))
plt.subplot(2,1,1)
plt.title("Relationship between companies and start_price",fontsize=40)
a=sns.barplot(data=dfps, y=dfps['start_price'],x=dfps['companies'])
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(21,9))
plt.subplot(2,1,2)
plt.title("Relationship between companies and final_price",fontsize=40)
a=sns.barplot(data=dfps1, y=dfps1['final_price'],x=dfps1['companies'])
plt.setp(a.get_xticklabels(), rotation=90)

plt.show()

The smallest 100 properties in terms of starting price, as shown in the bar plot, are associated with the "Anandavanam" company, which has the lowest starting prices and highest "APR praveens higheria "company in the Hyderabad real estate market.

The smallest 100 properties in terms of final price, as shown in the bar plot, are associated with the "seshadri richland farms" company, which has the highest final prices in the Hyderabad real estate market.

plt.figure(figsize=(11,8))
plt.subplot(2,1,1)
plt.title("Relationship between number_Floors and Start_price",fontsize=20)
sns.boxplot(data=df, y='start_price', x='number_floors')
plt.figure(figsize=(11,8))
plt.subplot(2,1,2)
plt.title("Relationship between number_floors and final_price",fontsize=20)
sns.boxplot(data=df, y='final_price', x='number_floors')

plt.show()

 From the box plot, it can be seen that there is a significant number of outliers in the data for properties with one floor. This suggests that the distribution of start prices for properties with one floor is more spread out compared to other values of the number of floors. This implies that there is a higher variance in start prices for properties with one floor, compared to other values of number of floors.

The plot indicates that properties with 0 floors have a higher number of outliers in terms of final price, compared to properties with other floor counts. This suggests that properties with 0 floors have a wider range of final prices, with some being significantly higher or lower than the average.

plt.figure(figsize=(11,8))
plt.subplot(2,1,1)
plt.title("Relationship between number_Floors and start_Area_sq_ft",fontsize=20)
sns.barplot(data=df, y='start_Area_sq_ft', x='number_floors')
plt.figure(figsize=(11,8))
plt.subplot(2,1,2)
plt.title("Relationship between number_floors and final_Area_sq_ft",fontsize=20)
sns.barplot(data=df, y='final_Area_sq_ft', x='number_floors')

plt.show()

The chart shows that properties with 1 floor have the highest start_Area_sq_ft and that properties with 0 floors have the highest final_Area_sq_ft.

plt.figure(figsize=(25,10),dpi=300)
plt.title("Relationship between locality and start_price",fontsize=40)
a=sns.boxplot(data=df, x='locality',y='start_price')
plt.setp(a.get_xticklabels(), rotation=90)
plt.figure(figsize=(25,10),dpi=300)
plt.title("Relationship between locality and floor_plan",fontsize=40)
b=sns.boxplot(data=df, x='locality', y='number_floors')
plt.setp(b.get_xticklabels(), rotation=90)
plt.figure(figsize=(21,8),dpi=300)
plt.title("Relationship between locality and Final",fontsize=40)
c=sns.boxplot(data=df, x='locality', y='final_price')
plt.setp(c.get_xticklabels(), rotation=90)
plt.show()

From the boxplots, it seems that the relationship between final_price, start_price, and number of floors with the locality is being displayed. The plots show that properties in "Madhapur" have a higher final_price, while properties in "Financial" Locality have a higher start_price. Additionally, properties in "APPA" with 25 floors have the highest number of floors.

sns.jointplot(data=df, x='bedrooms', y='final_price', kind='scatter')
plt.legend(title="Bedroom Count vs Final Price Scatterplot",fontsize=20,loc="best")
sns.jointplot(data=df, x='bedrooms', y='start_price',kind='scatter')
plt.legend(title="Bedroom Count vs start Price Scatterplot",fontsize=20,loc="best")
plt.show()

From the scatterplot, it appears that there is a positive correlation between the number of bedrooms and the final price as well as the start price. This means that as the number of bedrooms increases, the final price and start price tend to increase as well. This positive correlation suggests that properties with more bedrooms tend to have higher prices, both at the start and final stages of the selling process.

 

#### "Pairwise Relationships between Variables in the Dataset"


sns.pairplot(df)
plt.legend(title="Pairwise Relationships between Variables in the Dataset",fontsize=20,loc="upper right")
plt.show()

The plot shows the relationship between multiple variables in the dataset by plotting the scatterplot and histograms for each pair of variables. The diagonal plots are histograms for each variable, and the off-diagonal plots are scatterplots between the variables. The plot can help to identify relationships between variables, distributions of individual variables, and outliers in the data.

#The heatmap created using Seaborn (sns) is a plot that shows the correlation between multiple variables in a dataset.
sns.heatmap(data=df.corr(),annot=True,fmt = '.2f', linewidths=0.5, cmap='coolwarm', linecolor='white')
plt.title("plot that shows the correlation between multiple variable in a dataset",fontsize=20)
plt.show()

The heatmap provides a visual representation of the strength of the relationships between the variables, with darker squares indicating a stronger correlation. Positive correlations are represented by squares that are red, while negative correlations are represented by squares that are blue. The annotations show the correlation coefficients, with values close to 1 indicating a strong positive correlation, values close to -1 indicating a strong negative correlation, and values close to 0 indicating little to no correlation.

plt.figure(figsize=(20,10))

df.groupby(['bedrooms'])[['start_price','final_price']].max().plot(kind='bar')
plt.title("Relationship between bedrooms,start_price and final_price",fontsize=20)
plt.show()
plt.figure(figsize=(20,10))

df.groupby(['bedrooms'])[['start_Area_sq_ft','final_Area_sq_ft']].max().plot(kind='bar')
plt.title("Relationship between bedrooms,start_Area_sq_ft and final_Area_sq_ft",fontsize=20)
plt.show()


The bar plot indicates that there is a relationship between the number of bedrooms (1 BHK, 2 BHK, etc.), the starting price, and the final price. As the number of bedrooms increases, the starting price also increases, with the highest starting price observed for 5 BHKs. On the other hand, the final price shows the opposite trend, with the highest final price observed for 2 BHKs. The difference between the start price and the final price suggests that the final price is generally higher than the starting price.

plt.figure(figsize=(21,9))
df.groupby(df1['locality'])['start_price'].max().plot(kind='bar',color='b')
plt.title("Realatioship between start_price and locality",fontsize=40)
plt.show()
plt.figure(figsize=(21,9))
df.groupby(df4['locality'])['final_price'].max().plot(kind='bar',color='b')
plt.title("Realatioship between final_price and locality",fontsize=40)
plt.show()

From the above bar plot "FANCIAL" is having the heighest starting price for the flats and  "Gachibowli" is second place for the starting heighest price.

From the above bar plot "jogipet" is having the heighest final price for the flats and "madhapur" is second place for the final heighest price.

sns.violinplot(x='bedrooms', y='start_price', data=df)
plt.title("Realatioship between start_price and bedrooms",fontsize=20)
sns.catplot(x='bedrooms', y='final_price', kind='violin', data=df)
plt.title("Realatioship between final_price and bedrooms",fontsize=20)
plt.show()
plt.show()


A violin plot can effectively display the relationship between the three variables, final_price, start_price and bedrooms. The width of the violin shape represents the density of data points for each variable, with thicker sections indicating higher density. From the plot, you can observe the median and quartile values for each variable and understand the relationship between final_price and start_price, as well as the influence of bedrooms on final_price. In this case, if you see more outliers for 2BHK/plots compared to final_price, it may indicate that there is a higher variation in final_price for 2BHK properties compared to other bedroom sizes. Similarly, if you see more outliers for 5BHK/plots compared to start_price, it may suggest that 5BHK properties have a wider range of starting prices compared to other bedroom sizes.





