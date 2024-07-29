from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22})


def pp(df, exp):
    df_exp = df[df.exp == exp]
    df_pprint = (
        df_exp.assign(
            open_layer=lambda ddf: ddf.hook_type.map(
                lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x]).rolling(2).apply(lambda x: x[0] == 0 and x[1] == 0
                                                                              )
        )
            .assign(
            close_layer=lambda ddf: ddf.hook_type.map(
                lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x]).rolling(2).apply(lambda x: x[0] == 1 and x[1] == 1)
        )
            .assign(indent_level=lambda ddf: (ddf.open_layer.cumsum() - ddf.close_layer.cumsum()).fillna(0).map(int))
            .sort_values(by="call_idx")
            .assign(mem_diff=lambda ddf: ddf.mem_all.diff() // 2 ** 20)
    )
    pprint_lines = [
        f"{'    ' * row[1].indent_level}{row[1].layer_type} {row[1].hook_type}  {row[1].mem_diff or ''}"
        for row in df_pprint.iterrows()
    ]
    for x in pprint_lines:
        print(x)


def plot_mem(
        df,
        exps=None,
        normalize_call_idx=False,
        normalize_mem_all=True,
        filter_fwd=False,
        return_df=False,
        output_file=None,
        title=' ',
):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)

        if title != ' ':
            plt.title(str(title))
        plt.ylabel('memory usage / MB')

        plot.legend(loc='best')
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_


def plot_mem_by_time(
        df,
        exps=None,
        normalize_call_idx=False,
        normalize_mem_all=False,
        filter_fwd=False,
        return_df=False,
        output_file=None,
        title=' ',
):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
        df_.mem_all = df_.mem_all // 2 ** 20
        df_.timestamp = df_.timestamp - df_[df_.call_idx == df_.call_idx.min()].timestamp.iloc[0]

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='timestamp', y='mem_all', label=exp)

        if title != ' ':
            plt.title(str(title))
        plt.ylabel('memory usage / MB')

        plot.legend(loc='best')
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_