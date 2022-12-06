import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

def func(row):
    behaviour = row['behaviour']
    if behaviour == 'strategic1s':
        behaviour = 'one-sided strategic'
    return f"{int(100*row['p'])}% {behaviour}"

df['fancy'] = df.apply(func, axis=1)
order = df.groupby('method')['vse'].max().sort_values(ascending=False).keys()

fig, ax = plt.subplots(figsize=(10, 4))
plt.xlabel('Voter Satisfaction Efficiency (%)')
plt.xticks(np.arange(11)/10, [f'{int(v*10)}%' for v in range(11)])
plt.xlim(df['vse'].min() + df['vse'].max() - 1, 1)
sns.stripplot(
    df,
    x='vse',
    y='method',
    hue='fancy',
    jitter=False,
    size=8,
    order=order,
    ax=ax
)
plt.grid()
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.show()
