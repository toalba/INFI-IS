import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

#source: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
d = np.genfromtxt('data/london_weather.csv', delimiter=",", skip_header=1 )

dt =  d[:,0] #Datum mit folgendem Aufbau: 19790103 (3.Jänner 1979)
# Aufteilen in Tag, Monat, Jahr
day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')

# Check ob es funktioniert hat
print("Jahr:", np.unique(year, return_counts=True))
print("Monat", np.unique(month, return_counts=True))
print("Tag:", np.unique(day, return_counts=True))
print("Jahr MIN MAX" , np.min(year), np.max(year))

sun = d[:,2] # Sonnenstunden
print (sun)

#PLausibilitätscheck
print("Sun MIN MAX" , np.min(sun), np.max(sun))
plt.boxplot(sun)
#plt.show()

sun1979 = sun[year == 1979] #Holen der Sonnenstunden im Jahr 1979
sun2020 = sun[year == 2020]
plt.boxplot([sun1979, sun2020] ) #Gegenüberstellung der Sonnenstunden
plt.ylabel("Sonnenstunden")
plt.xticks([1,2],  ["1979","2020"])

#plt.show()


#########
# Temperatur
max_temp = d[:,4]
mean_temp = d[:,5]
min_temp = d[:,6]

temp_1979 = mean_temp[year == 1979]
temp_2019 = mean_temp[year == 2019]
temp_2000 = mean_temp[year == 2000]
temp_2010 = mean_temp[year == 2010]
plt.boxplot([temp_1979, temp_2019, temp_2000, temp_2010] ) #Gegenüberstellung der Temperaturen
plt.ylabel("Temperatur")
plt.xticks([1,2,3,4],  ["1979","2019", "2000", "2010"])
plt.show()

# Linechart for min, mean, max temp over the year 2019
temp_months = dt[year == 2019] % 10000 / 100
df = pd.DataFrame({'Temperatur': temp_months, 'Monate': temp_2019})
fig = px.line(df,y="Monate", x='Temperatur', title='Min Temp 2019', line_shape='linear')
fig.show()

# Linechart for min, mean, max temp over the year 2019
temp_max_2019 = max_temp[year == 2019]
temp_min_2019 = min_temp[year == 2019]
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp_months, y=temp_2019, mode='lines', name='mean temp'))
fig.add_trace(go.Scatter(x=temp_months, y=temp_max_2019, mode='lines', name='max temp'))
fig.add_trace(go.Scatter(x=temp_months, y=temp_min_2019, mode='lines', name='min temp'))
fig.update_layout(title='Temperatur 2019', xaxis_title='Monate', yaxis_title='Temperatur')
fig.show()

# In dem obrigen Diagramm ist zu erkennen, dass im Sommer eine Hoechsttemperatur von 37.9 Grad erreicht wird.
# Im Winter sinkt die Temperatur auf -5.7 Grad.

extremwerte = np.quantile(max_temp, [0.25, 0.75])

fig = go.Figure()
