import numpy as np

#1
a = np.arange(100, 201)
print(a)
b = np.arange(100, 201, 2)
print(b)
c = np.arange(100, 201, 0.5)
# Erzeuge 100 normalverteilte Zufallszahlen und 100 gleichverteilte Zufallszahlen
d = np.random.randn(100)
e = np.random.rand(100)
print(d)
print(e)
#2 mittelwert, median, min, max, standardabweichung
print(np.mean(e))
print(np.median(e))
print(np.min(e))
print(np.max(e))
print(np.std(e))
# Alle Zahlen mit 100 multiplizieren
print(a * 100)
# Alle Zahlen quadrieren
print(a ** 2)
# Alle Zahlen die größer als 0 sind
print(a[a > 0])