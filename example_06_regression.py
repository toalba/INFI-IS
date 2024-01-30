import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

# Einlesen des Datensatzes
# import bev_meldedaten
df = pd.read_excel('data/bev_meld.xlsx', header=0)


y7 = df.loc[df['month'] == 7][['year','mean_temp']]
y7g = y7.groupby('year').mean()
y7g.plot(title="Mittelwerte im Juli")

df_reg = pd.DataFrame({"years" : years, "mean_temp" : mean_temp})
df_reg = df_reg.astype({'years':'int'})

pred_years = np.arange(2030, 2160)

# Umbenennen der Spalten
base = ['Bezirk', 'Gemnr', 'Gemeinde']
years = df.columns[3:].astype(str)
base.extend('j' + years)
df.columns = base

model = sm.OLS.from_formula('mean_temp ~ years', data=df_reg).fit()
# Gesamtbevölkerung pro Jahr berechnen
total_population = df.groupby('year')['population'].sum()

# Grafische Darstellung der Gesamtbevölkerung
plt.figure(figsize=(10, 6))
total_population.plot(title="Gesamtbevölkerung pro Jahr")
plt.xlabel('Jahr')
plt.ylabel('Gesamtbevölkerung')
plt.show()


# Berechnung der prognostizierten Gesamtbevölkerung für das Jahr 2030 mit den Koeffizienten der Regressionsgeraden
pred_population_2030 = model.params[1] * 2030 + model.params[0]

print("Prognostizierte Gesamtbevölkerung für das Jahr 2030:", pred_population_2030)

# Prognose der Gesamtbevölkerung von 2030 bis 2100
pred_years = np.arange(2030, 2101)
pred_population = model.predict(pd.DataFrame({"years": pred_years}))

# Grafische Darstellung der prognostizierten Gesamtbevölkerung
plt.figure(figsize=(10, 6))
plt.plot(pred_years, pred_population, label='Prognose')
plt.xlabel('Jahr')
plt.ylabel('Gesamtbevölkerung')
plt.title('Prognose der Tiroler Gesamtbevölkerung (2030-2100)')
plt.legend()
plt.show()


# Filtern nach der Wohngemeinde (Beispiel: Innsbruck-Land)
wohngemeinde_df = df[df['Bezirk'] == 'Innsbruck-Land']

# Regressionsmodell für die Wohngemeinde erstellen
wohngemeinde_model = sm.OLS.from_formula('population ~ year', data=wohngemeinde_df).fit()

# Prognose der Bevölkerungsentwicklung in der Wohngemeinde bis zum Jahr 2100
pred_population_wohngemeinde = wohngemeinde_model.predict(pd.DataFrame({"year": pred_years}))

# Grafische Darstellung der Bevölkerungsentwicklung in der Wohngemeinde
plt.figure(figsize=(10, 6))
plt.plot(pred_years, pred_population_wohngemeinde, label='Prognose')
plt.xlabel('Jahr')
plt.ylabel('Bevölkerung')
plt.title('Prognose der Bevölkerungsentwicklung in der Wohngemeinde (2030-2100)')
plt.legend()
plt.show()


# Daten für die beiden Bezirke filtern (Beispiel: IL und RE)
bezirk_IL = df[df['Bezirk'] == 'IL']
bezirk_RE = df[df['Bezirk'] == 'RE']

# Regressionsmodelle für die beiden Bezirke erstellen
model_IL = sm.OLS.from_formula('population ~ year', data=bezirk_IL).fit()
model_RE = sm.OLS.from_formula('population ~ year', data=bezirk_RE).fit()

# Prognose der Bevölkerungsentwicklung in den beiden Bezirken bis zum Jahr 2100
pred_population_IL = model_IL.predict(pd.DataFrame({"year": pred_years}))
pred_population_RE = model_RE.predict(pd.DataFrame({"year": pred_years}))

# Grafische Darstellung der Bevölkerungsentwicklung in den beiden Bezirken
plt.figure(figsize=(10, 6))
plt.plot(pred_years, pred_population_IL, label='IL Prognose')
plt.plot(pred_years, pred_population_RE, label='RE Prognose')
plt.xlabel('Jahr')
plt.ylabel('Bevölkerung')
plt.title('Prognose der Bevölkerungsentwicklung in den Bezirken (2030-2100)')
plt.legend()
plt.show()
