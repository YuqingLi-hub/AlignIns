import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import os
if __name__ == "__main__":
    argpar = argparse.ArgumentParser()
    argpar.add_argument('--job', type=str, default='FLGMMTest00001', help='job name')
    
    args = argpar.parse_args()
    save_dir = f'outputs/FLGMM/Statistic{args.job}'
    
    log_file = f'outputs/R-QIM/{args.job}-out.txt'

    pattern = re.compile(
        r"agent (\d+) has mean ([\-\deE\.]+), and std ([\-\deE\.]+) after local training"
    )

    records = []
    iteration = 0

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                client_id = int(match.group(1))
                mean = float(match.group(2))
                std = float(match.group(3))
                records.append({"iter": iteration, "client": client_id,
                                "mean": mean, "std": std})
            # heuristic: every "local epoch 0" marks a new iteration
            if "local epoch 0" in line and "client: 0" in line:
                iteration += 1

    # Put into dataframe
    df = pd.DataFrame(records)
    print(df.head())

    from scipy.stats import ttest_ind

    means_clients01 = df[df["client"].isin([0,1])]["mean"]
    means_others    = df[~df["client"].isin([0,1])]["mean"]

    stds_clients01 = df[df["client"].isin([0,1])]["std"]
    stds_others    = df[~df["client"].isin([0,1])]["std"]

    mean_test = ttest_ind(means_clients01, means_others)
    std_test = ttest_ind(stds_clients01, stds_others)
    print("Mean difference test:", ttest_ind(means_clients01, means_others))
    print("Std difference test:", ttest_ind(stds_clients01, stds_others))

    # Std comparison
    import seaborn as sns

    sns.kdeplot(data=df[df["client"].isin([0,1])], x="mean", fill=True, label="client 0/1")
    sns.kdeplot(data=df[~df["client"].isin([0,1])], x="mean", fill=True, label="others")



    plt.title(f"Distribution of Mean with T test\n {mean_test}")
    plt.legend()
    plt.show()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/FLGMM_Means.png')

    
    sns.kdeplot(data=df[df["client"].isin([0,1])], x="std", fill=True, label="client 0/1")
    sns.kdeplot(data=df[~df["client"].isin([0,1])], x="std", fill=True, label="others")
    # sns.kdeplot(
    #     data=df[df["client"].isin([0, 1])], x="std", fill=True, label="client 0/1", common_norm=False
    # )
    # sns.kdeplot(
    #     data=df[~df["client"].isin([0, 1])], x="std", fill=True, label="others", common_norm=False
    # )

    plt.title(f"Distribution of Std with T test\n {std_test}")
    plt.legend()
    plt.show()


    plt.savefig(f'{save_dir}/FLGMM_Stds.png')


