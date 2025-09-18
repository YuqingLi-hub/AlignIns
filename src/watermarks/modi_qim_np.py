import numpy as np
import os
from datetime import date
date = date.today().strftime("%Y-%m-%d_m")
import sys
from collections import defaultdict
# from watermarks.tools import henon_map, plot_any
import torch
class QIM:
    def __init__(self, delta):
        # delta is the step size of quantization
        self.delta = delta
        # self.alpha = 0.51  # alpha is the weight of the original vector in the embedding
        self.fAlpha = ['quasi_periodic','CTBCS']  # alpha function type, can be 'linear', 'logistic', or 'cosine' 'CTBCS'
        # self.fAlpha = ['quasi_periodic','henon']  # alpha function type, can be 'linear', 'logistic', or 'cosine' 'CTBCS'
        self.r = 3.9
    def embed(self, x, m,alpha=0.51,k=0):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        # make x type float
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        x = x.astype(float)
        # state alpha
        if self.fAlpha[1] == 'henon':
            alpha = quanti(alpha,self.delta/10)
            alpha = quasi_periodic(alpha)
            k = quanti(k,self.delta/10)
            k = quasi_periodic(k)*2/5-0.2
            # print(alpha,k)
            # henon_points = np.array(henon_map(alpha,k,n=len(x)))
            # scale,k = henon_points[:,0]/6+0.75,henon_points[:,1]
            # print(scale.max(),scale.min())
        else:
            scale = self.alpha_func(alpha=alpha,n=len(x))

        # print(alpha)
        # print(-np.exp(-(alpha-0.5)**2 / 10**2)*0.5+1)
        # print(sigmoid(alpha))
        # print(np.where(scale<=0.5),np.where(scale>=1))
        # print(scale[np.where(scale<=0.5)][:10])
        # print(scale)
        d = self.delta
        # get d_0 and d_1 according to m
        # dm = (-1)**(m+1) * d/2.
        dm = m*d/2.
        q_mk = quanti((x-dm-k), d) + (dm + k)
        # dis = np.round((x-dm-k) / d) - (x-dm-k)/d
        # print('Theory Embedding distortion: $a*delta*(frac((s-d_m-k)/delta))', alpha*d*dis)
        # self.q_mk = q_mk
        self.x = x
        y = q_mk * scale + x * (1 - scale)
        return y
    
    def alpha_func(self, alpha,n=1000):
        r = self.r
        # eps = 1e-3
        alpha = quanti(alpha,self.delta/10)
        # print('Quantized alpha:', alpha)
        if self.fAlpha[0] =='sigmoid':
            x =sigmoid(alpha)
        elif self.fAlpha[0] =='quasi_periodic':
            x = quasi_periodic(alpha)
        else:
            x = bump(alpha)
        # x = np.exp(-(alpha-0.5)**2 / 10**2)
        # print(x)
        # print(x)
        # count_out_range = 0
        if self.fAlpha[1] == 'logistic':
            points = []
            for _ in range(n):
                # if x<=0.5:
                #     x = 0.5+x
                #     count_out_range+=1
                # elif x>=1:
                #     x = 1-eps
                #     count_out_range+=1
                points.append(x)
                x = logistic_map(x,r)
            # return np.clip(points, 0.5+eps, 1-eps)  # Ensure alpha is within [0, 1]
            points = np.array(points)*0.5+0.5
            # print(count_out_range)
            # print(np.where(points>=1), np.where(points<=0.5))
            # print(np.min(points), np.max(points))
            return np.array(points)
        elif self.fAlpha[1] == 'CTBCS':
            points = []
            for _ in range(n):
                points.append(x)
                x = CTBCS(x,beta=0.5)
            # return np.clip(points, 0.5+eps, 1-eps)  # Ensure alpha is within [0, 1]
            # print(np.min(points), np.max(points))
            points = np.array(points)*0.24+0.75
            # print(count_out_range)
            # print(np.where(points>=1), np.where(points<=0.5))
            # print(np.min(points), np.max(points))
            print(np.array(points).dtype)
            return np.array(points)
        
        elif self.fAlpha[1] == 'bump':
            
            return np.exp(-(alpha-0.5)**2 / 3**2)
        
        return alpha
    def detect(self, z,alpha=1,k=0,scale_delta=1):
        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """
        d = self.delta *scale_delta
        if not isinstance(z, np.ndarray):
            z = z.detach().cpu().numpy()
        if self.fAlpha[1] == 'henon':
            alpha = quanti(alpha,self.delta/10)
            alpha = quasi_periodic(alpha)
            k = quanti(k,self.delta/10)
            k = quasi_periodic(k)*2/5-0.2
            # print(alpha,k)
            # henon_points = np.array(henon_map(alpha,k,n=len(z)))
            # scale,k = henon_points[:,0]/6+0.75,henon_points[:,1]
            # print(len(np.where(scale<=0.5)[0]),len(np.where(scale>=1)[0]))
            # print(k.min(),k.max())
            # print(len(np.where(k<=-1)[0]),len(np.where(scale>=1)[0]))
            # print(scale.max(),scale.min())
            # print()
        else:
            scale = self.alpha_func(alpha=alpha,n=len(z))
        M_cls = 2.
        shape = z.shape
        
        # print(scale)
        z = z.flatten()
        z = z.astype(float)
        m_detected = np.zeros_like(z, dtype=float)
        dm_hat = (quanti((z-k),d/M_cls)+k)
        # self.dm_hat = dm_hat
        # self.m_c = self.selective_round(dm_hat/d)%1
        # self.y_dm_hat = abs(z-self.q_mk)
        z_hat = (z-scale * dm_hat)/ (1-scale)
        d_values = [0,d/2.]
        rough_m = np.round((self.selective_round((dm_hat-k)/d)%1)*2)
        m_detected = rough_m
        m_detected = m_detected.reshape(shape)
        return z_hat, m_detected.astype(int)
    
    def selective_round(self,x, threshold=0.99):
        frac = x%1
        if np.allclose(frac,0.5):
            frac = 0.5
        if np.allclose(frac,0):
            frac = 0
        return np.floor(x) + np.where((x % 1) >= threshold, 1, (frac))
    def random_msg(self, l):
        """
        returns: a random binary sequence of length l
        """
        return np.random.choice((0, 1), l)

def quanti(x, delta):
    """
    quantizes the input x with step size delta
    """
    # the delta*floor[x/delta]
    # so floor will increase the distortion
    return np.round(x / delta) * delta
def bump(alpha):
    return np.exp(-(alpha-0.5)**2 / 10**2)
def sigmoid(alpha):
    return 1/(1+np.exp(-alpha))
def quasi_periodic(t):
    return (((np.sin(t) + np.sin(np.sqrt(2) * t))+2)/4)**1.1
def logistic_map(x, r=3.9):
    return r * x * (1 - x)
def sine_map(x,r=3.9):
    return r * np.sin(np.pi * x)
def CTBCS(t,f1=logistic_map,f2=sine_map,beta=0.5):
    return np.cos(np.pi*f1(t)+f2(t,r=1-3.9/4)-beta)






def test_qim_1(delta=1,embedding_alpha=0.99,k=0,plot=False,test=False):
    """
    tests the embed and detect methods of class QIM
    """
    np.random.seed(42)
    l = 10000 # binary message length
    # delta = 1.0 # quantization step (use float for consistency)
    qim = QIM(delta)

    # x = np.random.uniform(-500, 500, l).astype(float) # host sample
    # x = np.random.randn(l)
    # x_torch = torch.zeros(l,dtype=torch.float32)
    points = []
    x = 0.7
    for _ in range(l):
        points.append(x)
        x = CTBCS(x,beta=0.5)
    # return np.clip(points, 0.5+eps, 1-eps)  # Ensure alpha is within [0, 1]
    # print(np.min(points), np.max(points))
    x_torch = torch.tensor(points)*1e-6
    # x_torch = torch.randn(l)*1e-5
    print(x_torch.dtype, type(x_torch))
    print(max(x_torch),min(x_torch))
    x = x_torch.cpu().numpy()
    print(x.dtype, type(x))
    # x = np.linspace(-5, 5, l).astype(float)  # host sample
    print('Original x (first 5):', x[:5])

    msg = qim.random_msg(l)
    print('Watermark Message (first 10):', msg[:10])

    # --- Crucial: Generate secret_k_sequence ONCE ---
    # This must be the same sequence used for both embedding and detection.
    np.random.seed(123) # Use a different seed for reproducibility of k sequence
    
    
    
    true_k = k
    true_f = 3
    # alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=embedding_alpha, delta_alpha=0.005, N=33, use_cos=True,f=true_f)
    # alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=0.7, delta_alpha=0.005, N=33, use_cos=False)
    # print('Alpha Pattern unique:', np.unique(alpha_pattern), len(np.unique(alpha_pattern)))
    # y_watermarked = qim.embed_alpha_pattern(x, msg, alpha_pattern, k=0.0)
    # good_z, good_msg = qim.detect(y_watermarked, alpha=alpha_pattern, k=0, scale_delta=1)
    # bad_alpha = qim.make_alpha_pattern(len(x), alpha_c=0.7, delta_alpha=0.005, N=33, use_cos=True,f=1000)
    # bad_z, bad_msg = qim.detect(y_watermarked, alpha=bad_alpha, k=0, scale_delta=1)
    # print('Alpha Pattern unique:', np.unique(bad_alpha), len(np.unique(bad_alpha)))
    # print(f"Detected Message Accuracy: {np.mean(msg == bad_msg):.4f}")
    # print(f'Recovery error when wrong: {np.mean(np.abs(bad_z - x)):.6f}')

    # Sweep detector α0
    # alpha0s = np.linspace(0.5, 0.9, 81)

    # scores = qim.sweep_alpha0(y_watermarked, alpha0s, k=0.0, carrier='exp', f=1.0)
    # print(np.argmin(scores), np.min(scores),alpha0s[np.argmin(scores)])
    # print(embedding_alpha)
    # print("Scores for α0 sweep:", scores[:10])  # Print first 10 scores for brevity
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(alpha0s, scores, marker='o', linestyle='-', markersize=4)
    # # plt.plot(alphas_to_test_detection, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    # # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    # plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    # plt.title(f'Sweep Score vs. Guessed Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
    # plt.xlabel('Detection Alpha ($a$)')
    # plt.ylabel('Score ($|\\hat{s} - s|$ mean)')
    # plt.ylim(0, 10)  # Optional: limit y for better visualization
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    # plt.savefig(f"./outputs/qim/{date}/score_alpha_{embedding_alpha}_delta_{delta}.png")
    # plt.close()




    # print('Watermarked y (first 5):', y_watermarked[:5])
    # print('Distortion: ',y_watermarked[:5]-x[:5])
    # initial_distortion = np.mean(np.abs(x - y_watermarked))
    # print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")
    # print(f"Detected Message Accuracy: {np.mean(msg == good_msg):.4f}")
    # print(f'Recovery error when all correct: {np.mean(np.abs(good_z - x)):.6f}')


    #######################################
    # # --- Step 1: Embed the watermark ONCE with a specific, fixed alpha ---
    # embedding_alpha = embedding_alpha # This is the "true" alpha used for embedding
    print(f"\n--- Embedding watermark with fixed alpha = {embedding_alpha}, delta = {delta}, k = {true_k} ---")
    print(f"")
    y_watermarked = qim.embed(x, msg, alpha=embedding_alpha, k=true_k)
    print('Watermarked y (first 5):', y_watermarked[:5])
    print('Distortion: ',y_watermarked[:5]-x[:5])
    initial_distortion = np.mean(np.abs(x - y_watermarked))
    good_z, good_msg = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=1)
    print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")
    print(f"Detected Message Accuracy: {np.mean(msg == good_msg):.4f}")
    print(f"Recovered?", np.allclose(good_z,x))
    print(x_torch.dtype)
    print(f"Recover torch?",torch.allclose(torch.tensor(good_z,dtype=x_torch.dtype).to(x_torch),x_torch))
    # print(qim.m_c[msg != good_msg])
    # # print((1-qim.m_c)[msg != good_msg])
    # # # print(qim.dm_hat[msg != good_msg]%1)
    # # print()
    # print(msg[msg != good_msg])
    # print(good_msg[msg != good_msg])
    print(f'Recovery error when all correct: {np.mean(np.abs(good_z - x)):.6f}')


    # --- Step 2: Loop through different 'a' values for DETECTION/RESTORATION ---
    # We are testing how well detection/restoration works if we GUESS 'a'
    # The 'y_watermarked' remains the same throughout this loop.
    # alphas_to_test_detection = np.arange(0.5, 1.00, 0.01) # Go up to 0.99 for range
    if test:
        recovery_errors = [] # This will store np.mean(np.abs(z_detected - x))
        message_accuracies = [] # This will store sum(msg==msg_detected)/len(msg)

        # # print("\n--- Testing Detection/Restoration with Varying Alphas ---")
        # alphas_to_test_detection =np.linspace(true_f-1, true_f+1, 200)
        # # guessed_alpha_c = [0.51,0.6,0.7,0.8,0.9,0.99]
        # guessed_alpha_c = np.linspace(-3,3, 100) # Guessed alpha_c values
        alphas_to_test_detection =  np.linspace(-7.5, 7.5,200)
        # # scale_alpha = np.linspace(-20, 20,2000)
        # # plot_any(scale_alpha, qim.alpha_func(scale_alpha), 
        # #          title='Alpha Function', xlabel='Alpha', ylabel='f(Alpha)', 
        # #          filename=f'alpha_function = beta*alpha*(1-alpha)+0.5')
        # # guess_alpha_f = defaultdict(list) # Store recovery errors for each guessed alpha_c
        for a_detect in alphas_to_test_detection:
        #     for g in guessed_alpha_c:
            z_detected, msg_detected = qim.detect(y_watermarked, alpha=a_detect, k=true_k)
            
        #         alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=g, delta_alpha=0.005, N=33, use_cos=True,f=a_detect)
        #         z_detected, msg_detected = qim.detect(y_watermarked, alpha=alpha_pattern, k=true_k)
        #         current_recovery_error = np.mean(np.abs(z_detected - x))
        #         guess_alpha_f[a_detect].append(current_recovery_error)
        #     # print(f"Testing with guessed alpha_c: {g}")
        #     # print(f"Attempting to detect/restore with detection alpha = {a_detect:.2f}")
        #     alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=guessed_alpha_c, delta_alpha=0.005, N=33, use_cos=True,f=a_detect)
        #     # print(np.unique(alpha_pattern))
        #     # alpha_pattern = a_detect
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=alpha_pattern, k=true_k)
        # secret_k_sequence = np.linspace(-10, 10, 1000)
        # for k in secret_k_sequence:
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=k)
        # # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.array([1,2,3,4,5])),axis=0)
        # # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.linspace(0.9, 1.1, 100)),axis=0)
        # # scale_delta = np.linspace(0.1, 5, 100)
        # # middle = 1
        # # range_scale = 0.2
        # # scale_delta = (np.linspace(middle-range_scale, middle+range_scale, 200))
        # # theory = []
        # # dm_hat = []
        # # for sd in scale_delta:
        # #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=sd)
        # #     theory.append(np.mean(np.abs(qim.q_mk - qim.dm_hat)*embedding_alpha/(1-embedding_alpha)))
        # #     dm_hat.append(np.mean(np.abs(qim.dm_hat)))
        #     # print(f"Attempting to detect/restore with detection alpha = {a_detect:.2f}")
        #     # if np.isclose(a_detect, 1, atol=0.005):
        #     #     recovery_errors.append(np.inf)
        #     #     continue  # Skip detection at alpha=1 to avoid division by zero in theory error
        #     # Use the SAME y_watermarked from the single embedding above
            
            
        #     # Calculate errors

            current_recovery_error = np.mean(np.abs(z_detected - x))
        #     # thery = np.mean(np.abs(qim.y_dm_hat*((a_detect-embedding_alpha)/((1-embedding_alpha)*(1-a_detect)))))

        #     # print(f"  Detected error (first 5): {current_recovery_error}")
        #     # print(f"  Theory error (first 5): {thery}")
        #     # print(np.isclose(current_recovery_error, thery, atol=1e-7, rtol=1e-7))
            current_message_accuracy = np.mean(msg == msg_detected) # Already normalized
        #     # print(f"  Detected message accuracy: {current_message_accuracy:.4f}")
            recovery_errors.append(current_recovery_error)
            message_accuracies.append(current_message_accuracy)
            
            # print(f"  Recovery Error (vs original x): {current_recovery_error:.8f}")
            # print(f"  Message Accuracy: {current_message_accuracy:.4f}")
            
            # Optional: Assert for the perfect case (only when a_detect matches embedding_alpha)
            # if np.isclose(a_detect, embedding_alpha, atol=0.005): # Use a small tolerance for float comparison
            #     # print("  --- Approaching True Alpha ---")
            #     assert np.allclose(x, z_detected, atol=1e-7, rtol=1e-7), \
            #         f"Host not perfectly restored at a_detect={a_detect}!"
                # assert np.all(msg == msg_detected), \
                #     f"Watermark not perfectly detected at a_detect={a_detect}!"
        # print('True f:', true_f, 'with guessed alpha:',guessed_alpha_c)
        # print('Guessed f:', alphas_to_test_detection[np.argmin(recovery_errors)],np.min(recovery_errors),np.mean(recovery_errors),embedding_alpha)
    if plot:
        # # --- Plotting Results ---
        from matplotlib import pyplot as plt
        # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
        # scaled_alpha = qim.alpha_func(embedding_alpha)
        # y = np.abs((alphas_to_test_detection - scaled_alpha) / ((1 - scaled_alpha) * (1 - alphas_to_test_detection)))
        # for a_detect in alphas_to_test_detection:
        if qim.fAlpha[0] == 'sigmoid':
            print('Using Sigmoid Alpha Function')
            print(sigmoid(embedding_alpha))
            plot_rec_mess(recovery_errors, message_accuracies, sigmoid(alphas_to_test_detection), sigmoid(embedding_alpha), delta)
        elif qim.fAlpha[0] == 'quasi_periodic':
            print('Using Quasi-Periodic Alpha Function')
            print((embedding_alpha))
            plot_rec_mess(recovery_errors, message_accuracies, (alphas_to_test_detection), (embedding_alpha), delta,filename=f"{qim.fAlpha[1]}_recovery_error_alpha{embedding_alpha}_quasi_periodic.png")
        else:
            print('Using Bump Alpha Function')
            print(bump(embedding_alpha))
            plot_rec_mess(recovery_errors, message_accuracies, bump(alphas_to_test_detection), bump(embedding_alpha), delta)
        # # Plot Recovery Error
        # plt.figure(figsize=(12, 6))
        # for g in alphas_to_test_detection:
        #     # if g % 0.01 <0.001 or g%true_f <0.1:  # Plot only every 0.01 or if close to embedding_alpha
        #         recovery_errors = guess_alpha_f[g]              #{alphas_to_test_detection[np.argmin(recovery_errors)]:.2f},{g:.2f}
        #         # print(recovery_errors[:10])
        #         plt.plot(guessed_alpha_c, recovery_errors, marker='o', linestyle='-', markersize=4,label=f'Guess f, alpha ={guessed_alpha_c[np.argmin(recovery_errors)]:.2f},{g:.2f},min_e={np.min(recovery_errors):.2f}')
        #     # plt.plot(alphas_to_test_detection, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
        #     # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
        #     # plt.axvline(x=np.sinc(embedding_alpha), color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        #     # plt.axhline(y=np.min(recovery_errors),color='r',linestyle='--', label=f'Min Recovery Error for {g}: {np.min(recovery_errors):.2f}')
        # # plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
        # plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True f={true_f}),alpha={embedding_alpha}')
        # plt.xlabel('Detection Alpha ($a$)')
        # # plt.xlabel('Guessed f ($f$)')
        # plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
        # plt.ylim(0, 1)  # Optional: limit y for better visualization
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
        # # plt.savefig(f"./outputs/qim/{date}/recovery_error_alpha_{embedding_alpha}_delta_{delta}.png")
        # plt.savefig(f"./outputs/qim/{date}/recovery_error_alpha_{embedding_alpha}_delta_{delta}_guess_alpha_{guessed_alpha_c[0]}-{guessed_alpha_c[-1]}_f_{true_f}regardingf.png")
        # plt.close()
        # plt.figure(figsize=(12, 6))
        # plt.plot(alphas_to_test_detection, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
        # plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        # plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha}),True f={true_f}')
        # plt.xlabel('Detection Alpha ($a$)')
        # plt.ylabel('Message Accuracy')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"./outputs/qim/{date}/message_accuracy_alpha_{embedding_alpha}.png")
        # plt.close()
    #     #################################################
    #     from matplotlib import pyplot as plt
    #     # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
    #     # y = np.abs((alphas_to_test_detection - embedding_alpha) / ((1 - embedding_alpha) * (1 - alphas_to_test_detection)))
    #     # Plot Recovery Error
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(secret_k_sequence, recovery_errors, linestyle='-', markersize=4)
    #     # plt.plot(secret_k_sequence, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    #     # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    #     plt.axvline(x=true_k, color='r', linestyle='--', label=f'True Embedding k ({true_k})')
    #     plt.title(f'Recovery Error vs. Detection secret k (Delta={delta}, True Embed k={true_k})')
    #     plt.xlabel('Detection secret k ($k$)')
    #     plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    #     # plt.ylim(0, 10)  # Optional: limit y for better visualization
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    #     plt.savefig(f"./outputs/qim/{date}/{qim.fAlpha[1]}recovery_error_k_{true_k}_delta_{delta}.png")
    #     plt.close()
    # # # Plot Message Accuracy
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(secret_k_sequence, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    #     plt.axvline(x=true_k, color='r', linestyle='--', label=f'True Embedding Alpha ({true_k})')
    #     plt.title(f'Message Detection Accuracy vs. Detection secret k (Delta={delta}, True Embed k={true_k})')
    #     plt.xlabel('Detection Alpha ($a$)')
    #     plt.ylabel('Message Accuracy')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"./outputs/qim/{date}/{qim.fAlpha[1]}message_accuracy_k_{true_k}_delta_{delta}.png")
    #     plt.close()

        # print("\n--- Testing complete. Check generated plots in ./outputs/ ---")
    #     ###########################################
    #     from matplotlib import pyplot as plt
    #     # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
    #     # y = np.abs((alphas_to_test_detection - embedding_alpha) / ((1 - embedding_alpha) * (1 - alphas_to_test_detection)))
    #     # Plot Recovery Error
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(scale_delta*delta, recovery_errors, marker='o', linestyle='-', markersize=4)
    #     plt.plot(delta, np.mean(np.abs(good_z-x)), marker="o", color="red")
    #     plt.plot(scale_delta*delta, theory, label=r"a/(1-a) Q_m,k(s) - d_m")
    #     # unscale_theory = ((1-embedding_alpha)/embedding_alpha)*np.array(theory)
    #     # plt.plot(scale_delta*delta, unscale_theory, label=r"$Q_m,k(s) - d_m$")
    #     plt.plot(scale_delta*delta,dm_hat, label=r"$\hat{d}_m$")
    #     # plt.plot(secret_k_sequence, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    #     # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    #     plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    #     plt.title(f'Recovery Error vs. Detection Delta={delta}')
    #     plt.xlabel('Detection Delta ($d$)')
    #     plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    #     plt.ylim(0, 10)  # Optional: limit y for better visualization
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    #     plt.savefig(f"./outputs/qim/{date}/recovery_error_delta_{delta}.png")
    #     plt.close()
    # # Plot Message Accuracy
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(scale_delta*delta, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    #     plt.plot(delta, np.mean(msg == good_msg), marker="o", color="red")
    #     plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    #     plt.title(f'Message Detection Accuracy vs. Detection Delta={delta}')
    #     plt.xlabel('Detection Delta ($d$)')
    #     plt.ylabel('Message Accuracy')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"./outputs/qim/{date}/message_accuracy_delta_{delta}.png")
    #     plt.close()

    #     print("\n--- Testing complete. Check generated plots in ./outputs/qim/ ---")

def plot_any(x,y, title, xlabel, ylabel, filename):
    """
    Generic plotting function for any x, y data.
    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-', markersize=4)
    plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5 Reference Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(0, 1)  # Optional: limit y for better visualization
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/{filename}.png")
    plt.close()
    
def plot_rec_mess(recovery_errors, message_accuracies, alphas_to_test_detection, embedding_alpha, delta,filename=f"recovery_error_alpha.png"):
    """
    Plot recovery errors and message accuracies.
    """
    from matplotlib import pyplot as plt
    # Plot Recovery Error
    plt.figure(figsize=(12, 6))
    plt.plot(alphas_to_test_detection, recovery_errors, marker='o',linestyle='-', markersize=4)
    plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    plt.title(f'Recovery Error vs. Detection Alpha Delta={delta} ,alpha={embedding_alpha}')
    plt.xlabel('Detection Alpha ($a$)')
    plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    plt.ylim(0, 1000)  # Optional: limit y for better visualization
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/{filename}")
    plt.close()

    # Plot Message Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(alphas_to_test_detection, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
    plt.xlabel('Detection Alpha ($a$)')
    plt.ylabel('Message Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./outputs/qim/{date}/message_accuracy_alpha_{embedding_alpha}.png")
    plt.close()
if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt

    # rng = np.random.default_rng(123)

    # # ---------- Basic utilities ----------
    # def Q_delta(u, Delta):
    #     # Nearest multiple of Delta
    #     return np.round(u / Delta) * Delta

    # def frac(u, Delta):
    #     # fractional part wrt Delta in [0, Delta)
    #     return (u % Delta)

    # # ---------- Embedding (beta-scaled R-QIM, binary m) ----------
    # def embed_beta_rqim(x, m, Delta=1.0, alpha=0.6, beta=4.0, k=0.0):
    #     """
    #     x: host vector
    #     m: {0,1} watermark bits (same shape as x)
    #     Delta: step
    #     alpha: mix weight (0<alpha<1)
    #     beta: scale used inside quantization domain
    #     k: shift (scalar or per-coordinate vector)
    #     """
    #     dm = (Delta/2.0) * m
    #     u  = beta * (x - (dm + k))
    #     q  = Q_delta(u, Delta)
    #     qp = (q / beta) + (dm + k)
    #     y  = (1.0 - alpha) * x + alpha * qp
    #     return y, qp  # return qp for diagnostics if you want

    # # ---------- One-step demix & bit decision for a candidate (alpha0, beta0, k0) ----------
    # def demix_and_bits(y, Delta=1.0, alpha0=0.6, beta0=4.0, k0=0.0):
    #     """
    #     Build candidate lattice from y, then demix to estimate x, and decide bits.
    #     """
    #     uy   = beta0 * (y - k0)
    #     qhat = Q_delta(uy, Delta)
    #     qph  = (qhat / beta0) + k0

    #     # demix (noiseless R-QIM inverse if alpha0 matches the embed alpha)
    #     xhat = (y - alpha0 * qph) / (1.0 - alpha0)

    #     # bit decision from coset of qph: near k0 => 0, near k0+Delta/2 => 1
    #     # turn coset into {0,1}
    #     coset = frac(qph - k0, Delta)  # in [0, Delta)
    #     mhat = np.round(2.0 * coset / Delta).astype(int)  # near 0 -> 0, near Delta/2 -> 1
    #     mhat = np.clip(mhat, 0, 1)
    #     return xhat, mhat, qph

    # # ---------- Detection score to sweep alpha0 ----------
    # def lattice_distance_score(y, Delta=1.0, alpha0=0.6, beta0=4.0, k0=0.0):
    #     """
    #     A self-consistency score: distance to lattice after one-step demix.
    #     Smaller is better; pick argmin over alpha0.
    #     """
    #     xhat, mhat, qph = demix_and_bits(y, Delta, alpha0, beta0, k0)

    #     # Recompute the "should-be" lattice point from xhat & decided bits (one refinement)
    #     dmhat = (Delta/2.0) * mhat
    #     u2 = beta0 * (xhat - (dmhat + k0))
    #     q2 = Q_delta(u2, Delta)

    #     # Mean squared distance to the lattice
    #     return np.mean((u2 - q2)**2)

    # # ---------- Demo ----------
    # n = 10000
    # Delta = 1.0
    # alpha_true = 0.62
    # beta_true  = 5.0

    # # Host and message
    # x = rng.normal(0.0, 1.0, size=n)
    # m = rng.integers(0, 2, size=n)  # 0/1 uniformly

    # # ----- Case A: public k (scalar 0.0) -> α-sweep reveals alpha -----
    # k_public = 0.0
    # y, _ = embed_beta_rqim(x, m, Delta=Delta, alpha=alpha_true, beta=beta_true, k=k_public)

    # alphas = np.linspace(0.05, 0.95, 181)  # sweep grid
    # scores_public = []
    # rec_err_public = []

    # for a0 in alphas:
    #     xhat, mhat, _ = demix_and_bits(y, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_public)
    #     scores_public.append(lattice_distance_score(y, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_public))
    #     rec_err_public.append(np.mean((xhat - x)**2))

    # alpha_hat_public = alphas[np.argmin(scores_public)]

    # print(f"[Public-k] argmin score alpha0 = {alpha_hat_public:.4f} (true alpha = {alpha_true})")
    # print(f"Public-k: min score = {np.min(scores_public):.6e}, MSE at min = {rec_err_public[np.argmin(scores_public)]:.6e}")

    # # ----- Case B: secret per-coordinate k_i -> α-sweep fails without the key -----
    # # Secret, per-coordinate k_i sampled from a PRNG with a (hidden) seed, limited to [0, Delta)
    # k_secret = rng.uniform(0.0, Delta, size=n)   # this is the "keyed" k_i
    # y_sec, _ = embed_beta_rqim(x, m, Delta=Delta, alpha=alpha_true, beta=beta_true, k=k_secret)

    # # Attacker wrongly assumes k0=0 (no key)
    # scores_secret_wrong = []
    # for a0 in alphas:
    #     scores_secret_wrong.append(lattice_distance_score(y_sec, Delta=Delta, alpha0=a0, beta0=beta_true, k0=0.0))

    # # (For reference) Legit receiver who knows the key recovers the sharp minimum
    # scores_secret_right = []
    # for a0 in alphas:
    #     scores_secret_right.append(lattice_distance_score(y_sec, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_secret))

    # alpha_hat_secret_right = alphas[np.argmin(scores_secret_right)]

    # print(f"[Secret-k, correct key] argmin score alpha0 = {alpha_hat_secret_right:.4f} (true alpha = {alpha_true})")
    # print(f"Secret-k: min score (correct-key) = {np.min(scores_secret_right):.6e}")

    # # ---------- Plots ----------
    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, scores_public, label="Score (public k=0)")
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Detection score vs α₀ (public k)")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("lattice-distance score (lower=better)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, scores_secret_wrong, label="Score (secret kᵢ, attacker assumes k₀=0)")
    # plt.plot(alphas, scores_secret_right, label="Score (secret kᵢ, correct key)")
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Detection score vs α₀ with secret per-coordinate kᵢ")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("lattice-distance score (lower=better)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, rec_err_public)
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Recovery MSE vs α₀ (public k)")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("MSE( x̂(α₀) , x )")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    import argparse
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--a', type=float, default=0.7,
                        help="number of training epochs")
    parser.add_argument('--d', type=float, default=1,
                        help="number of users: n")
    parser.add_argument('--k', type=float, default=0,
                        help="number of users: n")
    args = parser.parse_args()
    sys.exit(test_qim_1(delta=args.d,embedding_alpha=args.a,k=args.k,plot=True,test=True))