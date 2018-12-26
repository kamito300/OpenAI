import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('result.log')
df.plot(x=df.columns[0], y=df.columns[1])
plt.show()

