import matplotlib.pyplot as plt


def plot(df, col_to_plot, n_lags, error, save=True, save_file_name='plot', n_perc=None):
    plt.figure(figsize=(16, 2))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if n_perc is not None:
        df = df.iloc[int(1-n_perc*len(df)):]
    ax.plot(df.index, df[col_to_plot])
    ax.legend(col_to_plot)
    ax.set_title('<< {0} >> n_lags: {1} & error: {2:.6f}'.format(save_file_name, n_lags, error))
    # df.loc[:, col_to_plot].plot()
    if save:
        fig.savefig(f'./output/{save_file_name}.jpg', dpi=300)
        plt.close(fig)
    fig.show()

