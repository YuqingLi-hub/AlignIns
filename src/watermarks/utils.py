import numpy as np

class RQIM:
    def __init__(self, delta):
        self.delta = delta

    def quantize(self, x, delta):
        """Uniform quantizer."""
        return np.floor(x / delta) * delta

    def embed(self, s, m, alpha=0.5, k=0):
        # TODO embedding probabably wrong, distortion not correct
        """
        Embeds message m into signal s.
        s: signal array
        m: binary message array (same shape as s)
        alpha: embedding strength
        k: dithering key
        """
        d = self.delta
        # dm = m * d / 2  # dm = -Δ/4 if m=0, +Δ/4 if m=1
        dm = (-1) ** (m + 1) * d / 4  # dm = -Δ/4 if m=0, +Δ/4 if m=1
        q = self.quantize(s - dm - k, d) + dm + k
        s_rqim = alpha * q + (1 - alpha) * s
        return s_rqim

    def detect(self, y, alpha=0.5, k=0):
        """
        Recovers signal and message from received vector y.
        Returns: reconstructed signal s_hat and detected message m_hat.
        """
        d = self.delta
        M = 2
        delta_M = d / M
        z0 = self.embed(y, 0)
        z1 = self.embed(y, 1)
        d0 = np.abs(y - z0)
        d1 = np.abs(y - z1)
        # print(d0[:5],d1[:5])
        # Step 1: Estimate dm_hat using Eq. (11)
        q_y = self.quantize((y - k), delta_M) + k
        print((y-q_y)[:5])
        m_detected = np.zeros_like(y, dtype=float)
        z_detected = np.zeros_like(y, dtype=float)
        gen = zip(range(len(y)), d0, d1)
        for i, dd0, dd1 in gen:
            if dd0 < dd1:
                m_detected[i] = 0
                z_detected[i] = z0[i]
            else:
                m_detected[i] = 1
                z_detected[i] = z1[i]
        # m_hat = np.array([1 if i>0.5-i else 0 for i in y-q_y])
        # print(m_detected[:5],z_detected[:5])
        dm = (-1) ** (m_detected + 1) * d / 4 
        # print(d/4)
        # print(dm_hat[:5])
        # print((q_y/d)[:5])
        # # Step 2: Classify m_hat based on whether dm_hat is closer to -d/4 or +d/4
        # m_hat = np.where(dm_hat -d/4 > k, 1, 0)
        q_est = (y - (1-alpha)*y) / alpha  # initial guess
        # Snap q_est to nearest valid bin for given m
        q = self.quantize(q_est - dm - k, d) + dm + k
        # Recover s exactly
        s_hat = (y - alpha*q) / (1 - alpha)
        return s_hat, m_detected.astype(int)
    
# rqim = RQIM(delta=1.0)
    # s = np.random.randn(1000)
    # m = np.random.randint(0, 2, size=s.shape)
    # # print(m[:5])
    # print('x: ',s[:5])
    # alpha = 0.51
    # k = 0

    # # Embed
    # y = rqim.embed(s, m, alpha=alpha, k=k)

    # # Add noise (optional)
    # y_noisy = y + np.random.normal(0, 0.01, size=y.shape)
    # # y_noisy = y
    # # Detect
    # s_hat, m_hat = rqim.detect(y_noisy, alpha=alpha, k=k)
    # print((y_noisy-s_hat)[:5])
    # # Evaluation
    # print("Message recovery accuracy:", np.mean(m_hat == m))
    # print("Signal reconstruction error (MSE):", np.mean((s_hat - s) ** 2))