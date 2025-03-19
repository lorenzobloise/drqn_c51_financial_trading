import matplotlib.pyplot as plt
import os
import json
import codecs

def plot_tests(path):
    tests = [test for test in os.listdir(path) if test != '.DS_Store']
    num_tests = len(tests)
    for t in range(num_tests):
        fig = plt.figure(figsize=(12, 12))
        file = os.path.join(path, f'test_{t + 1}', 'args.json')
        with codecs.open(file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        row_title = f"gamma={args.get('gamma')}, lr={args.get('lr')}, memory={args.get('replay_memory_size')}, batch={args.get('batch_size')}"
        portfolio = os.path.join(path, f'test_{t + 1}', 'portfolio')
        stocks = [stock for stock in os.listdir(portfolio) if stock != '.DS_Store' and not stock.endswith('png')]
        title_ax = fig.add_subplot(1, 1, 1, frame_on=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.spines['top'].set_visible(False)
        title_ax.spines['bottom'].set_visible(False)
        title_ax.spines['left'].set_visible(False)
        title_ax.spines['right'].set_visible(False)
        title_ax.set_title(row_title, fontsize=12, fontweight='bold', pad=20)
        for i, stock in enumerate(stocks):
            curr_stock = stock[0:3]
            ax = fig.add_subplot(len(stocks), 1, i + 1)
            with open(os.path.join(portfolio, f'{curr_stock}_portfolio.json')) as f:
                test_results = json.load(f)
            ax.plot(test_results)
            ax.axes.get_xaxis().set_ticks([0, len(test_results)])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
            ax.set_title(curr_stock)
        plt.savefig(portfolio+f'/results_test_{t + 1}.png', dpi=300, bbox_inches='tight')
        plt.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.close()

def plot_multiple_tests(tests):
    fig = plt.figure(figsize=(12, 12))
    stocks = ['ABB','AMZ']
    for i, stock in enumerate(stocks):
        ax = fig.add_subplot(len(stocks), 1, i + 1)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.set_title(stock)
        for path in tests.keys():
            portfolio = os.path.join(path, 'portfolio')
            with open(os.path.join(portfolio, f'{stock}_portfolio.json')) as f:
                test_results = json.load(f)
            ax.plot(test_results, label=tests.get(path))
        ax.axhline(y=0, color='r', linestyle='--')
        ax.axes.get_xaxis().set_ticks([0, len(test_results)])
        ax.legend()
    plt.savefig("./plots/plot.png", dpi=300, bbox_inches='tight')
    plt.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
    plt.close()