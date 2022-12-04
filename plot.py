import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
fig, ax = plt.subplots()
plt.xlim(right=1)
plt.xlabel('Voter Satisfaction Efficiency (%)')
# ticks = np.arange(11)/10
# labels = [f'{int(v*10)}%' for v in range(11)]
# plt.xticks(ticks, labels)
sns.stripplot(df, x='vse', y='method', jitter=False, ax=ax)
plt.show()
