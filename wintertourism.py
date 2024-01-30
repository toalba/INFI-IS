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
df["Gemeinde"] = df["Gemeinde"].str.strip()


#print(df.describe())
#print(tabulate(df, headers=df.columns))

#print(df.Bezirk)
# get all years from gemeinde innsbruck
i = df[df.Gemeinde == 'Innsbruck']
# stelle den zeitlichen Verlauf als Punktdiagramm da
y_i = i.iloc[0,3:].astype(int)
x_i = y_i.index
# plotly express scatter plot
#fig = px.scatter(x=x_i, y=y_i, title='Innsbruck').show()

# get all years from bezirk innsbruck-land and sum the values
i_l = df[df.Bezirk == 'IL']
# sum a
i_l_sum = i_l.iloc[0:,3:].sum(axis=0)
i_l_x = i_l_sum.index
# plotly express scatter plot
#fig = px.scatter(x=i_l_x, y=i_l_sum, title='Innsbruck-Land').show()

#3
#df['min'] = df.iloc[:,3:].min(axis=1)
#df['max'] = df.iloc[:,3:].max(axis=1)
#df['mean'] = df.iloc[:,3:].mean(axis=1)
#df['range'] = df['max'] - df['min']

ges_tourists_per_year = df.iloc[:,3:].sum(axis=1)
ges_tourists_overall = ges_tourists_per_year.sum()
# zusammenfassung touristen pro bezirk
df_bez = df.groupby('Bezirk').sum()
df_bez.plot.bar(title='Touristen pro Bezirk')
#plt.show()

# zusammenfassung touristen in innbruck
df_inn = df[df.Bezirk == 'I']
df_inn.plot.bar(title='Touristen in Innsbruck')
#plt.show()

# import bev_meldedaten
df_bev = pd.read_excel('data/bev_meld.xlsx', header=0)
base = ['Bezirk','Gemnr','Gemeinde']
years = df_bev.columns[3:].astype(str)
base.extend('j' + years)
df_bev = df_bev[1:]
df_bev.columns = base
both = pd.merge(df_bev, df, how='inner', on = 'Gemnr')
# touristen(x) pro einwohner(j)
both['ratio18'] = both['x2018'] / both['j2018']
print(both.describe())