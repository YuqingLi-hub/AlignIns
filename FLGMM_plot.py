import pandas as pd
import matplotlib.pyplot as plt
import re



with open(f'outputs/R-QIM/FLGMMTest00001-out.txt', 'r') as file:
    lines = file.readlines()
# Regex pattern to match lines with agent, mean, and std values
pattern = re.compile(r'agent (\\d+) has mean ([\\d.e-]+), and std ([\\d.e-]+) after local training')

# Lists to hold the extracted data
agent_ids = []
iterations = []
means = []
stds = []

# A dictionary to keep track of the iteration count for each agent
iteration_counts = {}

for line in lines:
    match = pattern.match(line.strip())
    if match:
        agent_id = int(match.group(1))
        mean_val = float(match.group(2))
        std_val = float(match.group(3))

        # Update iteration count for the current agent
        if agent_id not in iteration_counts:
            iteration_counts[agent_id] = 0
        else:
            iteration_counts[agent_id] += 1
        
        agent_ids.append(agent_id)
        iterations.append(iteration_counts[agent_id])
        means.append(mean_val)
        stds.append(std_val)

# Create a DataFrame from the extracted data
df = pd.DataFrame({
    'agent': agent_ids,
    'iteration': iterations,
    'mean': means,
    'std': stds
})

# Get unique agents
unique_agents = df['agent'].unique()

# Create a plot for each agent
for agent_id in unique_agents:
    agent_df = df[df['agent'] == agent_id]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot mean values
    ax1.plot(agent_df['iteration'], agent_df['mean'], marker='o', linestyle='-')
    ax1.set_title(f'Mean over Iterations for Agent {agent_id}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean')
    ax1.grid(True)

    # Plot std values
    ax2.plot(agent_df['iteration'], agent_df['std'], marker='o', linestyle='-', color='red')
    ax2.set_title(f'Standard Deviation over Iterations for Agent {agent_id}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'agent_{agent_id}_metrics.png')