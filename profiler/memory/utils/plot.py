from matplotlib import pyplot as plt


def plot_mem_by_time(
        df,
        output_file=None,
        title=' ',
):
    title_font_size = 8
    tick_font_size = 8
    label_size = 8
    linewidth=1

    df['mem_all'] = df['mem_all'] / 2 ** 30
    mem_all = df['mem_all'].to_list()

    timestamp = df['timestamp'].to_list()
    firsttime = timestamp[0]
    timestamp = [x-firsttime for x in timestamp]

    figure, axes = plt.subplots()
    x_range = [0,8]
    y_range = [0,6]
    axes.set_xlim(x_range)
    axes.set_ylim(y_range)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    plt.plot(timestamp, mem_all, linewidth=1, color='brown')

    plt.xlabel('Time (s)', fontsize=title_font_size)
    plt.ylabel('Mem. usage (GB)', fontsize=title_font_size)
    plt.title(title,  fontsize=title_font_size, y=1.05, x=0.4)
    # Time v.s. Batch Size

    # Calculate Pearson correlation coefficient


    # plt.legend(fontsize=label_size, markerfirst=False, borderpad=0.2, loc='upper left')
    figure.set_size_inches(1.5, 1)
    figure.set_dpi(100)
    plt.savefig(output_file, bbox_inches='tight')