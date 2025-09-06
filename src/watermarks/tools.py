import numpy as np
from datetime import date
date = date.today().strftime("%Y-%m-%d_m")
import os
from matplotlib import pyplot as plt
from collections import defaultdict

def henon_attractor(x, y, a=1.4, b=0.3):
	'''Computes the next step in the Henon 
	map for arguments x, y with kwargs a and
	b as constants.
	'''
	x_next = 1 - a * x ** 2 + y
	y_next = b * x
	return x_next, y_next

def henon_map(x,y,a=1.4,b=0.3,n=1000):
    '''
    Generates the Henon map for n iterations
    starting from initial values x and y with
    kwargs a and b as constants.
    Returns a list of tuples (x, y) representing
    the points in the Henon attractor.
    '''
    points = []
    for _ in range(n):
        points.append((x, y))
        x, y = henon_attractor(x, y, a, b)
        # if not np.isfinite(x) or not np.isfinite(y):
        #     break
    return points

def logistic_map(x, r=3., n=1000):
    """
    Generates the logistic map for n iterations
    starting from initial value x with parameter r.
    Returns a list of values representing the logistic map.
    """
    points = []
    for _ in range(n):
        points.append(x)
        x = r * x * (1 - x)
    return points

def plot_any(x,y, title, xlabel, ylabel, filename,std=None,norm=None):
    """
    Generic plotting function for any x, y data.
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-', markersize=4)
    if std is not None and norm is not None:
        plt.plot(x, norm+std, marker='x', linestyle='--', color='orange', label='Mean + Std Dev')
        plt.plot(x, norm, marker='x', linestyle='--', color='green', label='Mean')
        plt.plot(x, norm-std, marker='x', linestyle='--', color='orange', label='Mean - Std Dev')
    # plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5 Reference Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(0, 0.1)  # Optional: limit y for better visualization
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/{filename}.png")
    plt.close()
def logistic_pattern(x, r=3.9, n=1000):
    points = []
    for _ in range(n):
        points.append(x)
        x = logistic_map(x,r=r)
    return points

def logistic_map(x, r=3.9):
    return r * x * (1 - x)
def sine_map(x,r=3.9):
    return r * np.sin(np.pi * x)
def CTBCS(t,f1=logistic_map,f2=sine_map,beta=0.5):
    return np.cos(np.pi*f1(t)+f2(t,r=1-3.9/4)-beta)
def CTBCS_map(x,beta=0.5,n=1000):
    points = []
    for _ in range(n):
        points.append(x)
        x = CTBCS(x,beta=beta)
    return points
if __name__ == "__main__":
    # initial_x = 0.5
    # iterations = 1000
    # points = CTBCS_map(initial_x,beta=0.5, n=iterations)
    # # print(count_out_range)
    # # print(np.where(points>=1), np.where(points<=0.5))
    # print(np.min(points), np.max(points))
    # plot_any(np.arange(len(points)), points, title="CTBCS map", xlabel="Iterations", ylabel="Value", filename=f"CTBCS_map_beta=0.5")
    # test_x = np.linspace(initial_x-0.01,initial_x+0.01, 1000)
    # error = []
    # for test_x_val in test_x:
    #     # for test_y_val in test_y:
    #     guess_p = CTBCS_map(test_x_val,n=iterations)
    #     error.append(np.mean(np.abs(np.array(guess_p) - np.array(points))))
    # # print("Error computed for test points:", error.values())
    # print( "with error", min(error))
    # plot_any(test_x,error, title="CTBCS error with small change", xlabel="X-axis", ylabel="Y-axis", filename="CTBCS_error_beta=0.5")
    
    ##########################################
    # # Example usage
    initial_x = 0.7
    initial_y = 0.
    iterations = 1000
    henon_points = henon_map(initial_x, initial_y, n=iterations)
    print(f"Generated {len(henon_points)} points in the Henon attractor.",np.max(henon_points, axis=0),np.min(henon_points, axis=0))
    test_x = np.linspace(initial_x-0.01,initial_x+0.01, 1000)
    # test_y = np.linspace(0.0099, 0.11111, 100)
    print(np.array(henon_points).shape)
    plot_any(np.arange(len(henon_points)), henon_points, title="Henon map", xlabel="Iterations", ylabel="Value", filename=f"Henon_map_a=1.4_b=0.3_initx={initial_x}_inity={initial_y}")

    # error = {}
    # for test_x_val in test_x:
    #     # for test_y_val in test_y:
    #     test_y_val = initial_y
    #     guess_p = henon_map(test_x_val, test_y_val)
    #     error[test_x_val,test_y_val] = np.mean(np.abs(np.array(guess_p) - np.array(henon_points)))
    # # print("Error computed for test points:", error.values())
    # print( "with error", min(error.values()))
    # plot_any(test_x,error.values(), title="Henon error with small change", xlabel="X-axis", ylabel="Y-axis", filename="henon_attractor")
##########################################
#     initial_x = 0.8
#     iterations = 1000
#     r=3.9
# #     rs = np.linspace(2.5, 3.3, 100)  # Range of r values for logistic map
# #     # rs = [1.5,2,2.5,3.,3.4,3.5]  # Example r values for logistic map
# #     # rs = [3.]
#     test_range = 0.01
# #     total_errors_for_r = []
# #     for r in rs:
#     logistic_points = logistic_pattern(initial_x,r=r, n=iterations)
# #         # print(f"Generated {len(logistic_points)} points in the Logistic map.",np.std(logistic_points, axis=0),np.mean(logistic_points, axis=0),r)
#     test_x = np.linspace(initial_x-test_range,initial_x+test_range, 1000)
# #         # test_y = np.linspace(0.0099, 0.11111, 100)

#     error = []
#     for test_x_val in test_x:
#         # for test_y_val in test_y:
#         guess_p = logistic_pattern(test_x_val,r=r, n=iterations)
#         error.append(np.mean(np.abs(np.array(guess_p) - np.array(logistic_points))))
#     plot_any(np.arange(len(logistic_points)), logistic_points, title="Logistic map", xlabel="Iterations", ylabel="Value", filename=f"Logistic_map_r={r}_initx={initial_x}")
#         # print("Error computed for test points:", error.values())
#         total_errors_for_r.append([min(error.values()),np.average(np.array(list(error.values()))), np.std(np.array(list(error.values())))])
#         # print( "with error", min(error.values()),np.average(np.array(list(error.values()))), np.std(np.array(list(error.values()))))
#         # plot_any(test_x,error.values(), title="Logistic error with small change", xlabel="X-axis", ylabel="Y-axis", filename=f"logistic_error_{r}")
#     # print(np.array(total_errors_for_r).shape)
#     total_errors_for_r = np.array(total_errors_for_r)
#     plot_any(rs, total_errors_for_r[:,0], title="Logistic error with small change", xlabel="r values", ylabel="Error", filename=f"logistic_error_min_r_{rs[0]},{rs[-1]}_test_range={test_range}<initial-x_{initial_x}",std=total_errors_for_r[:,2], norm=total_errors_for_r[:,1])

# ###########################################
#     rs = np.linspace(3.5, 3.9, 10) 
#     ranges_for_r = defaultdict(list)
#     init_xs = np.linspace(0.51, 0.99, 10)
#     for init_x in init_xs:
#         for r in rs:
#             logistic_points = logistic_map(initial_x,r=r, n=iterations)
#             min_val = np.min(logistic_points)
#             max_val = np.max(logistic_points)
#             q25, q75 = np.percentile(logistic_points, [25, 75])  # majority distribution
#             std = np.std(logistic_points)
#             mean = np.mean(logistic_points)
#             # print(f"std{std}, mean{mean}, min{min_val}, max{max_val}, q25{q25}, q75{q75} for init_x={init_x:.2f} and r={r:.2f}")
#             ranges_for_r[init_x].append([min_val, max_val, q25, q75,mean,std])
#         print('min',np.min(np.array(ranges_for_r[init_x]),axis=0))
#         print('max',np.max(np.array(ranges_for_r[init_x]),axis=0))
#         print('std',np.std(np.array(ranges_for_r[init_x]),axis=0))
#     # ranges_for_r = np.array(ranges_for_r)
#     # print(ranges_for_r.shape)
#     plt.figure(figsize=(12, 6))

#     for init_x, values in ranges_for_r.items():
#         r_range = np.array(values)  # shape: (len(rs), 4) [min, max, q25, q75]
#         norm = r_range[:,4]
#         std = r_range[:,5]
#         # plt.plot(rs, norm+std, marker='x', linestyle='--', color='orange', label='Mean + Std Dev')
#         plt.plot(rs, norm, marker='x', linestyle='--', color='green', label='Mean')
#         # plt.plot(rs, norm-std, marker='x', linestyle='--', color='orange', label='Mean - Std Dev')
#         plt.fill_between(rs, norm-std, norm+std, alpha=0.15, label=f'ini_x={init_x:.2f} IQR')
#         # # Shade between q25 and q75
#         # plt.fill_between(rs, r_range[:,2], r_range[:,3], alpha=0.3, label=f'ini_x={init_x:.2f} IQR')
        
#         # # Median line = midpoint between q25 and q75
#         # median = (r_range[:,2] + r_range[:,3]) / 2
#         # plt.plot(rs, median, label=f'ini_x={init_x:.2f} Median', linewidth=1.5)
        
#         # # # Optionally min/max bounds (thin lines)
#         # plt.plot(rs, r_range[:,0], '--', alpha=0.5, linewidth=0.8, label=f'ini_x={init_x:.2f} Min')
#         # plt.plot(rs, r_range[:,1], '--', alpha=0.5, linewidth=0.8, label=f'ini_x={init_x:.2f} Max')
#     # plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5 Reference Line')
#     plt.title('Logistic Map Ranges with Varying Initial x and r')
#     plt.xlabel('r values')
#     plt.ylabel('Logistic Map Values')
#     plt.grid(True)

#     # Move legend outside to avoid clutter
#     plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)

#     plt.tight_layout()
#     os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
#     plt.savefig(f"./outputs/qim/{date}/results_range_of_init_x.png", bbox_inches="tight")
#     plt.close()

    # # plt.figure(figsize=(12, 6))
    # # for init_x, values in ranges_for_r.items():
    # #     r_range = np.array(ranges_for_r[init_x])
    # #     # print(r_range.shape, init_x)
    # #     plt.plot(rs, r_range[:,0], label=f'Min')
    # #     plt.plot(rs, r_range[:,1], label=f'Max')
    # #     plt.plot(rs, r_range[:,2], label=f'q25')
    # #     plt.plot(rs, r_range[:,3], label=f'q75')
    # # # plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5 Reference Line')
    # # plt.title('Logistic Map Ranges with Varying Initial x and r')
    # # plt.xlabel('r values')
    # # plt.ylabel('Range of Logistic Map Values')
    # # # plt.ylim(0, 0.1)  # Optional: limit y for better visualization
    # # # plt.legend()
    # # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    # # plt.grid(True)
    # # plt.tight_layout()
    # # os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    # # plt.savefig(f"./outputs/qim/{date}/results_range_of_init_x.png")
    # # plt.close()

    # import numpy as np
    # import matplotlib.pyplot as plt

    # def logistic_map(x, r, n_iter=1000, discard=200):
    #     """Run logistic map iterations and return trajectory after discarding transients."""
    #     xs = []
    #     for _ in range(n_iter):
    #         x = r * x * (1 - x)
    #         xs.append(x)
    #     return np.array(xs[discard:])  # discard transients

    # # Parameters to test
    # r_values = [3.2, 3.9]  # 3.2 ~ periodic regime, 3.9 ~ chaotic regime
    # init_conditions = [0.1, 0.5, 0.9]  # slightly different initial x

    # results = {}

    # for r in r_values:
    #     stats = []
    #     for x0 in init_conditions:
    #         traj = logistic_map(x0, r)
    #         stats.append([traj.min(), traj.max(), traj.mean(), traj.std()])
    #     results[r] = np.array(stats)

    # # Print statistics
    # for r in r_values:
    #     print(f"\n=== r = {r} ===")
    #     for i, s in enumerate(results[r]):
    #         print(f"init x={init_conditions[i]} → min={s[0]:.4f}, max={s[1]:.4f}, mean={s[2]:.4f}, std={s[3]:.4f}")
