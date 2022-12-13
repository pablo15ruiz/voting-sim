import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vse(df):
    def func(row):
        b = row['behaviour']
        b = b if b != 'strategic1s' else 'one-sided strategic'
        return f"{int(100*row['p'])}% {b}"

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

def plot_cycle(df):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(ncols=2, wspace=0)
    ax1, ax2 = gs.subplots(sharey=True)
    ax1.set_ylabel('Cycle frequency')
    ticks = np.array([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
    plt.yticks(ticks, [f'{100*v}%' for v in ticks])
    sns.lineplot(df, x='n_vot', y='cycle_freq', hue='model', legend=True, ax=ax1)
    ax1.set_xlabel('Voters')
    sns.lineplot(df, x='n_cand', y='cycle_freq', hue='model', legend= False, ax=ax2)
    ax2.set_xlabel('Candidates')
    plt.show()

def plot_cycle_dim(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_ylabel('Cycle frequency')
    ticks = np.array([0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016])
    plt.yticks(ticks, [f'{100*v}%' for v in ticks])
    sns.lineplot(df, x='n_dim', y='cycle_freq', hue='model', legend=True, ax=ax)
    plt.xlabel('Dimensions')
    plt.show()

# df = pd.read_csv('real_cycles.csv')
# df = df[df['model'] != 'ImpartialCulture']
# df = df[df['n_vot'] >= 1000]
# plot_cycle(df)
# df = pd.read_csv('cycles_dim.csv')
# plot_cycle_dim(df)
methods = ['FPTP', 'Borda', 'IRV', 'Approval', 'Score5', 'STAR5', '3-2-15', 'MajorityJudgment5']
df = pd.read_csv('all_vse.csv')
df = df[df['p'].isin([0.5, 1])]
df = df[df['method'].isin(methods)]
print(df)
plot_vse(df)
