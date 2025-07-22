import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = './results/'
FIG_DIR = '_Figures'
TASK = []
MODELS = ['bc', 'gcpc', 'gail', 'rnn-gail', 'drail', 'gail-mk2', 'drail-mk2']
COLORS = ['orange', 'brown', 'green', 'yellow', 'purple', 'red', 'blue']
LINES = ['-.', '-.', '-', '-', '-', '-', '-']
MARKERS = ['o', '^', 'v', '<', '>', 's', 'x']

NUM_ENV_STEPS = {'Maze':20e6, 'Pick':25e6, 'Push':5e6, 'Hand':5e6, 'Ant':10e6}
WINDOW_SIZE = 4e-6

def moving_average(x, window_size=10):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

def main(task, num_env_steps, window_size):
    plt.figure(figsize=(5, 4))
    plt.rcParams["font.size"] = 16
    plt.xlabel("Step")
    plt.ylabel("Goal Completion (%)")
    plt.xlim(0, num_env_steps)
    plt.ylim(0, 105)
    plt.grid(linestyle=':')

    xticks = np.linspace(0, num_env_steps, 6, dtype=int)
    xtick_labels = ['0' if x == 0 else f'{int(x/1e6)}M' for x in xticks]

    for idx, model in enumerate(MODELS):
        result_file = os.path.join(RESULTS_DIR, task, f'{model}.csv')
        
        if not os.path.exists(result_file):
            continue

        df = pd.read_csv(result_file)
        
        steps = np.array(df['Step'].values)

        values = df.iloc[:, 1:]*100

        color = COLORS[idx % len(COLORS)]
        line = LINES[idx % len(LINES)]
        marker = MARKERS[idx % len(MARKERS)]

        if model in ['bc', 'gcpc']:
            steps = xticks
            values = pd.DataFrame([values.iloc[-1].values] * len(steps), columns=values.columns)

        mean, min_, max_ = values.iloc[:, 0], values.iloc[:, 1], values.iloc[:, 2]

        if not model in ['bc', 'gcpc']:
            mean = moving_average(mean, window_size=window_size)
            min_ = moving_average(min_, window_size=window_size)
            max_ = moving_average(max_, window_size=window_size)
            steps = steps[len(steps) - len(mean):]

        plt.plot(steps, mean, linestyle=line, color=color)
        plt.fill_between(steps, min_, max_, alpha=0.2, color=color)

        marker_idx = np.linspace(0, len(steps)-1, 6, dtype=int)
        plt.plot(steps[marker_idx], mean[marker_idx], linestyle='None', marker=marker, color=color)

    plt.xticks(ticks=xticks)
    plt.gca().set_xticklabels(xtick_labels)

    plt.axhline(y=100, color='black', linestyle='--', linewidth=1.5)
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, FIG_DIR, f'{task}.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    fig_dir = os.path.join(RESULTS_DIR, FIG_DIR)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for dir in os.listdir(RESULTS_DIR):
        if os.listdir(os.path.join(RESULTS_DIR, dir)):
            if dir != FIG_DIR:
                num_env_steps = NUM_ENV_STEPS.get(dir.split('-')[0])
                window_size = int(num_env_steps* WINDOW_SIZE)
                main(dir, num_env_steps, window_size)
