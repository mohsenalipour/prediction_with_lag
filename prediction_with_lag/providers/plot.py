import matplotlib.pyplot as plt


def plot(df, col_to_plot, n_lags, save=True):
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(df.index, df[col_to_plot])
    ax.legend(col_to_plot)
    ax.set_title(f'n_lags: {n_lags}')
    # df.loc[:, col_to_plot].plot()
    if save:
        fig.savefig('./output/plot.jpg', dpi=300,  bbox_inches='tight')
        plt.close(fig)
    fig.show()

