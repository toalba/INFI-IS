import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel('data/Zeitreihe-Winter-2022092012.xlsx',header=2)
base = ['Bezirk','Gemnr','Gemeinde']
years = df.columns[3:].astype(str)
base.extend('x' + years)
df = df[1:]
df.columns = base
for i in range(1,len(df.Gemeinde)):
    #See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    #df.Gemeinde[i] = df.Gemeinde[i].strip()
    #/home/toalba/Projects/INFI-IS/wintertourism.py:13: SettingWithCopyWarning: 
    #A value is trying to be set on a copy of a slice from a DataFrame
    
    df.Gemeinde[i] = df.Gemeinde[i].strip()


#print(df.describe())
#print(tabulate(df, headers=df.columns))

#print(df.Bezirk)
# get all years from gemeinde innsbruck
i = df[df.Gemeinde == 'Innsbruck']
# stelle den zeitlichen Verlauf als Punktdiagramm da
y_i = i.iloc[0,3:].astype(int)
x_i = y_i.index
# plotly express scatter plot
fig = px.scatter(x=x_i, y=y_i, title='Innsbruck').show()

# get all years from bezirk innsbruck-land and sum the values
i_l = df[df.Bezirk == 'IL']
# sum a
i_l_sum = i_l.iloc[0:,3:].sum(axis=0)
i_l_x = i_l_sum.index
# plotly express scatter plot
fig = px.scatter(x=i_l_x, y=i_l_sum, title='Innsbruck-Land').show()
df['min'] = df.iloc[:,3:].min(axis=1)
df['max'] = df.iloc[:,3:].max(axis=1)
df['mean'] = df.iloc[:,3:].mean(axis=1)
df['range'] = df['max'] - df['min']
