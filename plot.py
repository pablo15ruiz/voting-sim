import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vse(df, save=False):
    df = df.query('vot == 1500 & cand == 5 & p.isin([0.5, 1])')
    order = df.groupby('meth')['vse'].max().sort_values(ascending=False).keys()
    hue_order = ['100% honesto', '50% estratégico unilateral', '50% estratégico', '100% estratégico unilateral', '100% estratégico']
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.stripplot(df, x='vse', y='meth', hue='fancy', jitter=False, size=8, order=order, hue_order=hue_order, ax=ax)
    plt.xlabel('% Eficiencia de la Satisfacción de los Votantes (VSE)')
    plt.ylabel(' ')
    plt.xticks(np.arange(11)/10, [f'{int(v*10)}%' for v in range(11)])
    plt.xlim(df['vse'].min() + df['vse'].max() - 1, 1)
    plt.grid()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))

    if save:
        plt.savefig('Plots/vse.png')
        plt.savefig('Plots/vse_hd.png', dpi=300)
    else:
        plt.show()

def plot_vse_p(df, save=False):
    df = df.query('vot == 1500 & cand == 5')
    df = df[~df['meth'].isin(['FPTP', 'Borda', 'IRV'])]
    order = df.groupby('meth')['vse'].max().sort_values(ascending=False).keys()

    df.loc[df['beh'] == 'Honesto', 'p'] = 0
    df['p'] = df.apply(lambda x: f"{int(100*x['p'])}%", axis=1)
    df2 = df.copy(deep=True)
    df2.loc[df2['beh'] == 'Honesto', 'beh'] = 'Estratégico unilateral'
    df.loc[df['beh'] == 'Honesto', 'beh'] = 'Estratégico'
    df = df[df['beh'] == 'Estratégico']
    df2 = df2[df2['beh'] == 'Estratégico unilateral']

    # df = df[df['p'] <= 0.5]
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10.5, 9))
    fig.subplots_adjust(hspace=0.35)
    sns.pointplot(df, x='vse', y='meth', hue='p', linestyles='dotted', order=order, scale=0.8, ax=ax1)
    sns.pointplot(df2, x='vse', y='meth', hue='p', linestyles='dotted', order=order, scale=0.8, ax=ax2)
    ax1.set_xlabel('% Eficiencia de la Satisfacción de los Votantes (VSE)')
    ax1.set_ylabel(' ')
    ax2.set_xlabel('% Eficiencia de la Satisfacción de los Votantes (VSE)')
    ax2.set_ylabel(' ')
    ticks = np.arange(0.91, 1., 0.01)
    ax1.set_xticks(ticks, [f'{int(100*v)}%' for v in ticks])
    ticks = np.arange(0.65, 1.05, 0.05)
    ax2.set_xticks(ticks, [f'{int(100*v)}%' for v in ticks])

    ax1.grid()
    ax2.grid()
    ax2.legend([], [], frameon=False)

    ax1.set_title('Estratégico')
    ax2.set_title('Estratégico unilateral')

    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width * 0.93, pos.height])
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0, pos.width * 0.93, pos.height])
    ax1.legend(loc='center right', bbox_to_anchor=(1.145, -0.15))

    if save:
        plt.savefig('Plots/vse_p.png')
        plt.savefig('Plots/vse_p_hd.png', dpi=300)
    else:
        plt.show()

def plot_vse_cand_honest(df, save=False):
    df = df[~df['meth'].isin(['FPTP', 'Borda', 'IRV'])]
    df = df.query('vot == 1500 & p == 1')
    df = df[df['beh'] == 'Honesto']
    hue_order = ['Smith/Rango', 'STAR', 'Rango', '3-2-1', 'MJ', 'Aprobatorio']

    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.pointplot(df, x='cand', y='vse', hue='meth', scale=0.8, linestyles='dotted', hue_order=hue_order)
    plt.xlabel('Candidatos')
    plt.ylabel('% VSE')
    ticks = np.arange(0.97, 0.996, 0.005)
    labels = ['97%', '97,5%', '98%', '98,5%', '99%', '99,5%']
    plt.yticks(ticks, labels)
    plt.grid()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.22, 0.5))
    plt.title('100% honesto')

    if save:
        plt.savefig('Plots/vse_cand_honest.png')
        plt.savefig('Plots/vse_cand_honest_hd.png', dpi=300)
    else:
        plt.show()

def plot_vse_cand_strat(df, save=False):
    df = df[~df['meth'].isin(['FPTP', 'Borda', 'IRV'])]
    df = df.query('vot == 1500 & p == 1')
    df = df[df['beh'] == 'Estratégico']
    hue_order = ['Smith/Rango', 'STAR', 'Rango', '3-2-1', 'MJ', 'Aprobatorio']

    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.pointplot(df, x='cand', y='vse', hue='meth', scale=0.8, linestyles='dotted', hue_order=hue_order)
    plt.xlabel('Candidatos')
    plt.ylabel('% VSE')
    ticks = np.arange(0.86, 0.98, 0.02)
    plt.yticks(ticks, [f'{int(100*v)}%' for v in ticks])
    plt.grid()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.22, 0.5))
    plt.title('100% estratégico')

    if save:
        plt.savefig('Plots/vse_cand_strat.png')
        plt.savefig('Plots/vse_cand_strat_hd.png', dpi=300)
    else:
        plt.show()

def plot_vse_diff(df, save=False):
    df = df[~df['meth'].isin(['FPTP', 'Borda', 'IRV'])]
    df = df[(df['vot'] == 1500) & (df['cand'] == 5) & (df['p'] == 1)]
    data = pd.DataFrame(df['meth'].unique(), columns=['meth'])
    data['diff'] = df[df['beh'] == 'Honesto']['vse'].to_numpy()
    data['beh'] = 'Estratégico'
    data['diff'] -= df[df['beh'] == 'Estratégico']['vse'].to_numpy()
    data2 = pd.DataFrame(df['meth'].unique(), columns=['meth'])
    data2['diff'] = df[df['beh'] == 'Honesto']['vse'].to_numpy()
    data2['beh'] = 'Estratégico unilateral'
    data2['diff'] -= df[df['beh'] == 'Estratégico unilateral']['vse'].to_numpy()
    data = pd.concat([data, data2])
    data.rename(columns={'beh': 'Comportamiento'}, inplace=True)

    order = ['Smith/Rango', 'STAR', 'Rango', '3-2-1', 'MJ', 'Aprobatorio']
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.stripplot(data, x='diff', y='meth', hue='Comportamiento', jitter=False, size=8, ax=ax, order=order)
    plt.grid()
    ticks = np.arange(0.05, 0.4, 0.05)
    plt.xscale('logit')
    plt.xticks(ticks, [f'{int(100*v)}%' for v in ticks])
    plt.xlabel('% Pérdida de VSE respecto 100% honesto')
    plt.ylabel(' ')

    if save:
        plt.savefig('Plots/vse_diff.png')
        plt.savefig('Plots/vse_diff_hd.png', dpi=300)
    else:
        plt.show()

def filter_df(df):
    df.loc[df['beh'] == 'honest', 'beh'] = 'Honesto'
    df.loc[df['beh'] == 'strategic', 'beh'] = 'Estratégico'
    df.loc[df['beh'] == 'one-sided strategic', 'beh'] = 'Estratégico unilateral'
    df.loc[df['meth'] == 'Score(5)', 'meth'] = 'Rango'
    df.loc[df['meth'] == 'STAR(5)', 'meth'] = 'STAR'
    df.loc[df['meth'] == 'MajorityJudgment(5)', 'meth'] = 'MJ'
    df.loc[df['meth'] == 'SmithScore(5)', 'meth'] = 'Smith/Rango'
    df.loc[df['meth'] == 'Approval', 'meth'] = 'Aprobatorio'
    ordinal = ['FPTP', 'Borda', 'IRV']
    # cardinal = ['Rango', 'Aprobatorio', 'STAR', 'Smith/Rango', 'MJ', '3-2-1']
    df['type'] = 'cardinal'
    df.loc[df['meth'].isin(ordinal), 'type'] = 'ordinal'
    df['fancy'] = df.apply(lambda x: f"{int(100*x['p'])}% {x['beh'].lower()}", axis=1)
    return df


df = filter_df(pd.read_csv('vse.csv'))

plot_vse(df.copy(deep=True))
plot_vse_p(df.copy(deep=True))
plot_vse_cand_honest(df.copy(deep=True))
plot_vse_cand_strat(df.copy(deep=True))
plot_vse_diff(df.copy(deep=True))
