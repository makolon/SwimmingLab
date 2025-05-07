import pandas as pd
import matplotlib.pyplot as plt
import glob


def load_and_aggregate(csv_pattern: str):
    """
    Load multiple CSV files matching the pattern,
    aggregate 'Train/mean_reward' across seeds per Step,
    and return a DataFrame with mean and std for each Step.
    """
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files match pattern: {csv_pattern}")

    frames = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        if 'Step' not in df.columns or 'Train/mean_reward' not in df.columns:
            raise KeyError(f"Required columns not found in {filepath}")
        df = df[['Step', 'Train/mean_reward']].copy()
        df.set_index('Step', inplace=True)
        frames.append(df)

    combined = pd.concat(frames, axis=1)
    combined.columns = [f"seed{i}" for i in range(len(frames))]

    aggregated = pd.DataFrame({
        'mean': combined.mean(axis=1),
        'std': combined.std(axis=1),
    })
    aggregated.index.name = 'Step'
    return aggregated


def plot_compare_rewards(
    lee_pattern: str,
    direct_pattern: str,
    figsize=(8, 5)
):
    """
    Plot and compare mean reward and variance between
    LeeVelController-based RL and direct RL across multiple seeds.
    """
    # Load and aggregate data for each setup
    lee_data = load_and_aggregate(lee_pattern)
    direct_data = load_and_aggregate(direct_pattern)

    plt.figure(figsize=figsize)
    # Plot LeeVelController RL results
    plt.plot(
        lee_data.index, lee_data['mean'],
        label='LeeVelController RL', linewidth=2
    )
    plt.fill_between(
        lee_data.index,
        lee_data['mean'] - lee_data['std'],
        lee_data['mean'] + lee_data['std'],
        alpha=0.2
    )

    # Plot Direct RL results
    plt.plot(
        direct_data.index, direct_data['mean'],
        label='Direct RL', linewidth=2
    )
    plt.fill_between(
        direct_data.index,
        direct_data['mean'] - direct_data['std'],
        direct_data['mean'] + direct_data['std'],
        alpha=0.2
    )

    plt.xlabel('Step')
    plt.ylabel('Train/mean_reward')
    plt.title('Comparison: LeeVelController RL vs Direct RL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage, adjust file patterns as needed
    plot_compare_rewards(
        lee_pattern="wandb_exports/seed_*_lee.csv",
        direct_pattern="wandb_exports/seed_*_direct.csv"
    )
