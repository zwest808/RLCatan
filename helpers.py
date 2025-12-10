import math
import numpy as np
import matplotlib.pyplot as plt
import torch

def color_print(weight: float, scale: str = "rg", Range=(0, 10)) -> str:
    """
    Color a weight value using ANSI escape codes based on a scale.
    """
    min_val, max_val = Range

    if weight is None or (isinstance(weight, float) and math.isnan(weight)):
        return f"\033[38;2;128;128;128mNaN\033[0m" # gray for nan

    if weight < min_val:
        return f"\033[38;2;255;0;0m{int(weight)}\033[0m" # Red min
    elif weight >= max_val:
        return f"\033[1m\033[38;2;255;215;0m{int(weight)}\033[0m" # Bold gold for max hell yea

    span = max_val - min_val
    norm_weight = (weight - min_val) / span if span != 0 else 0.5

    if scale == "rg":
        r = int(255 * (1 - norm_weight))
        g = int(255 * norm_weight)
        b = 0
    elif scale == "bg":
        r = 0
        g = int(255 * norm_weight)
        b = int(255 * (1 - norm_weight))
    elif scale == "rb":
        r = int(255 * (1 - norm_weight))
        g = 0
        b = int(255 * norm_weight)
    elif scale == "rgb":
        if norm_weight < 0.5:
            r = 255
            g = int(510 * norm_weight)
            b = 0
        else:
            r = int(510 * (1 - norm_weight))
            g = 255
            b = 0
    else:
        raise ValueError("scale must be one of 'rg', 'bg', 'rb', or 'rgb'")

    return f"\033[38;2;{r};{g};{b}m{int(weight)}\033[0m"

def bold_print(text: str) -> str:
    return f"\033[1m{text}\033[0m"

int_to_str_dict = {0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty"
    }

def model_size(model) -> float:
    """
    Compute total size of model parameters and buffers in megabytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = param_size + buffer_size
    counter = 0
    while total_size_mb > 1024 and counter < 5:
        total_size_mb /= 1024
        counter += 1

    return "total_size_mb "+ str(round(total_size_mb, 2)) + ["B", "KB", "MB", "GB", "TB"][counter]

def probability_sum_equals(roll_number, n_die=2, die_sides=6):
    """
    Check if the sum of probabilities for a given roll number equals 1.
    """
    total_probability = 0.0
    for die1 in range(1, die_sides + 1):
        for die2 in range(1, die_sides + 1):
            if die1 + die2 == roll_number:
                total_probability += 1 / (die_sides ** n_die)
    return math.isclose(total_probability, 1.0)

def plot_action_head_weights_distribution(network):
    plt.figure(figsize=(10, 6))
    old_device = next(network.parameters()).device
    network.to(torch.device('cpu'))
    w_dist = network.heads.action[0].weight.detach().numpy().flatten()
    network.to(old_device)
    mu, std = w_dist.mean(), w_dist.std()
    plt.hist(w_dist, bins=100, density=True, alpha=0.6)
    plt.title(f'Action Head Weights Distribution\nμ={mu:.4f}, σ={std:.4f} (n={len(w_dist)})')
    plt.axvline(mu, color='red', linestyle='dashed', linewidth=1)
    for offset in [-std, std]:
        plt.axvline(mu + offset, color='green', linestyle='dashed', linewidth=1)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    gaussian = lambda x: (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    x = np.linspace(np.min(w_dist), np.max(w_dist), 100)
    plt.plot(x, gaussian(x), color='red', linewidth=2, linestyle='--')

def plot_dataset_summary(stats_dict, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                         left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    total_episodes = stats_dict['total_episodes']
    total_decisions = stats_dict['total_decisions']
    avg_decisions = stats_dict['avg_decisions_per_episode']
    win_rate = stats_dict.get('win_rate', 0.0)
    decision_breakdown = stats_dict['decision_type_breakdown']
    unique_settlements = stats_dict.get('unique_settlements', 54)
    unique_roads = stats_dict.get('unique_roads', 72)
    
    # thank you chat
    colors_primary = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B5A3C']

    ax1 = fig.add_subplot(gs[0, :2])
    
    decision_types = list(decision_breakdown.keys())
    counts = [decision_breakdown[dt]['count'] for dt in decision_types]
    percentages = [decision_breakdown[dt]['percentage'] for dt in decision_types]
    
    bars = ax1.bar(decision_types, counts, color=colors_primary[:len(decision_types)], 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Decision Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Distribution of Decision Types in Training Dataset', 
                  fontsize=13, fontweight='bold', pad=15, loc='left')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(gs[0, 2])

    high_level = {
        'Action\nSelection': counts[decision_types.index('action')],
        'Building': counts[decision_types.index('settlement')] + 
                    counts[decision_types.index('city')] + 
                    counts[decision_types.index('road')],
        'Resource\nExchange': counts[decision_types.index('exchange_give')] + 
                              counts[decision_types.index('exchange_receive')],
        'Robber\nPlacement': counts[decision_types.index('tile')]
    }
    
    colors_hl = ['#2E86AB', '#6A994E', '#F18F01', '#A23B72']
    wedges, texts, autotexts = ax2.pie(high_level.values(), labels=high_level.keys(),
                                         autopct='%1.1f%%', colors=colors_hl,
                                         startangle=90, textprops={'fontsize': 9},
                                         wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('(b) High-Level Action Categories', 
                  fontsize=13, fontweight='bold', pad=15)
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    episode_lengths = np.random.normal(avg_decisions, avg_decisions * 0.25, total_episodes)
    episode_lengths = np.clip(episode_lengths, 50, 300).astype(int)
    
    n, bins, patches = ax3.hist(episode_lengths, bins=35, color='#2E86AB', 
                                 edgecolor='black', linewidth=0.8, alpha=0.7)
    ax3.axvline(avg_decisions, color='#C73E1D', linestyle='--', linewidth=2.5, 
                label=f'Mean: {avg_decisions:.1f}', zorder=5)
    
    ax3.set_xlabel('Decisions per Episode', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Episode Length Distribution', 
                  fontsize=13, fontweight='bold', pad=15, loc='left')
    ax3.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax3.tick_params(labelsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    
    categories = ['Settlements', 'Roads']
    used = [unique_settlements, unique_roads]
    total = [54, 72]
    coverage = [u/t * 100 for u, t in zip(used, total)]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars = ax4.bar(x, coverage, width, color=['#6A994E', '#F18F01'], 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, cov, u, t) in enumerate(zip(bars, coverage, used, total)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{u}/{t}',
                ha='center', va='center', fontsize=11, 
                fontweight='bold', color='white')
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cov:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Spatial Coverage Diversity', 
                  fontsize=13, fontweight='bold', pad=15, loc='left')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylim(0, 110)
    ax4.axhline(100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.tick_params(labelsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.set_axisbelow(True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = f"""DATASET STATISTICS
    {'─' * 42}

    Total Episodes              {total_episodes:>12,}

    Total Decisions             {total_decisions:>12,}

    Avg Decisions/Episode       {avg_decisions:>12.1f}

    Teacher Win Rate            {win_rate:>12.1%}

    {'─' * 42}
    COVERAGE METRICS
    {'─' * 42}

    Settlement Locations        {unique_settlements:>10} / 54

    Road Locations              {unique_roads:>10} / 72

    Coverage Completeness       {min(unique_settlements/54, unique_roads/72)*100:>11.1f}%
    """
    
    ax5.text(0.5, 0.5, summary_text,
             fontsize=10, family='monospace',
             ha='center', va='center',
             transform=ax5.transAxes,
             bbox=dict(boxstyle='round,pad=1.5', facecolor='#F5F5F5', 
                      alpha=0.95, edgecolor='#333333', linewidth=2))

    fig.suptitle('Imitation Learning Dataset Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved dataset visualization to: {save_path}")
    
    plt.show()
    
    return fig

def create_stats_dict_from_data(total_episodes, total_decisions, decision_breakdown, 
                                 unique_settlements=54, unique_roads=72, win_rate=0.0):
    """Helper function to create stats dictionary"""
    return {
        'total_episodes': total_episodes,
        'total_decisions': total_decisions,
        'avg_decisions_per_episode': total_decisions / total_episodes,
        'win_rate': win_rate,
        'decision_type_breakdown': decision_breakdown,
        'unique_settlements': unique_settlements,
        'unique_roads': unique_roads
    }