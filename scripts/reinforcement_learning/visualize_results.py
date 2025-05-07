import pandas as pd
import matplotlib.pyplot as plt
import glob


def load_and_aggregate(csv_pattern: str) -> pd.DataFrame:
    """
    Load multiple CSV files matching the pattern,
    extract the appropriate 'Train/mean_reward' column (ignoring __MIN/__MAX),
    compute per-step mean and standard deviation across seeds,
    and return a DataFrame indexed by Step with columns ['mean', 'std'].
    """
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files match pattern: {csv_pattern}")

    frames = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        # find the column for Train/mean_reward (ignore __MIN, __MAX)
        reward_cols = [
            col for col in df.columns
            if 'Train/mean_reward' in col and not col.endswith('__MIN') and not col.endswith('__MAX')
        ]
        if 'Step' not in df.columns or len(reward_cols) != 1:
            raise KeyError(
                f"Could not find a single 'Train/mean_reward' column in {filepath}; "
                f"found: {reward_cols}"
            )
        reward_col = reward_cols[0]
        # select and rename
        sub = df[['Step', reward_col]].rename(columns={reward_col: 'Train/mean_reward'})
        sub.set_index('Step', inplace=True)
        frames.append(sub)

    # concatenate columns (each seed as one column)
    combined = pd.concat(frames, axis=1)
    combined.columns = [f"seed{i}" for i in range(len(frames))]

    # compute mean and std across seeds
    aggregated = pd.DataFrame({
        'mean': combined.mean(axis=1),
        'std': combined.std(axis=1),
    })
    aggregated.index.name = 'Step'
    return aggregated


def plot_compare_rewards(
    lee_pattern: str,
    rotor_pattern: str,
    task_name1: str = "ObstacleNavLeeVel",
    task_name2: str = "ObstacleNavRotor",
    figsize: tuple = (8, 5)
):
    """
    Plot Train/mean_reward mean and variance for two experimental setups.
    """
    lee_data = load_and_aggregate(lee_pattern)
    rotor_data = load_and_aggregate(rotor_pattern)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot task1 results
    ax.plot(
        lee_data.index, lee_data['mean'],
        label=task_name1, linewidth=2
    )
    ax.fill_between(
        lee_data.index,
        lee_data['mean'] - lee_data['std'],
        lee_data['mean'] + lee_data['std'],
        alpha=0.3
    )

    # Plot task2 results
    ax.plot(
        rotor_data.index, rotor_data['mean'],
        label=task_name2, linewidth=2
    )
    ax.fill_between(
        rotor_data.index,
        rotor_data['mean'] - rotor_data['std'],
        rotor_data['mean'] + rotor_data['std'],
        alpha=0.3
    )

    ax.set_xlabel('Step')
    ax.set_ylabel('Train/mean_reward')
    ax.set_title(f'Comparison: {task_name1} vs {task_name2}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("obstacle_navigation_comparison.png")


if __name__ == "__main__":
    # Example usage with your directory patterns
    plot_compare_rewards(
        lee_pattern="./results/ObstacleNavLeeVel/ObstacleNav-HummingBird-LeeVel-seed-*.csv",
        rotor_pattern="./results/ObstacleNavRotor/ObstacleNav-HummingBird-Rotor-seed-*.csv",
        task_name1="ObstacleNavLeeVel",
        task_name2="ObstacleNavRotor",
    )
