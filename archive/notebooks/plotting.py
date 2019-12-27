"""
Short Plotting Routine to Plot Pandas Dataframes by Column Label

1. Takes list of dateframes to compare multiple trials
2. Takes list of y-variables to combine on 1 plot
3. Legend location and y-axis limits can be customized

Written by Patrick Coady (pat-coady.github.io)
"""
import matplotlib.pyplot as plt


def df_plot(dfs, x, ys, ylim=None, xlim=None, legend_loc='best'):
    """ Plot y vs. x curves from pandas dataframe(s)

    Args:
        dfs: list of pandas dataframes
        x: str column label for x variable
        ys: list of str column labels for y variable(s)
        ylim: tuple to override automatic y-axis limits
        xlim: tuple to override automatic x-axis limits
        legend_loc: str to override automatic legend placement:
            'upper left', 'lower left', 'lower right' , 'right' ,
            'center left', 'center right', 'lower center',
            'upper center', and 'center'
    """
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    for df, name in dfs:
        if '_' in name:
            name = ",".join(name.split('_')[0:2])
        for y in ys:
            plt.plot(df[x], df[y], linewidth=1,
                     label=name)
    plt.xlabel(x.replace('_', ''))
    plt.ylabel(y.replace('_', ''))
    plt.legend(loc=legend_loc)
    plt.show()
