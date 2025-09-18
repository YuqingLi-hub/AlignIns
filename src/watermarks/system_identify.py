from modi_qim_np import QIM, quasi_periodic, quanti, logistic_map,CTBCS
from tools import henon_map
import numpy as np
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Legendre
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.general_estimators import NARX
# from catboost import CatBoostRegressor
from sysidentpy.basis_function import Fourier
if __name__ == "__main__":
    # np.random.seed(42)
    l = 10000 # binary message length
    delta = 1.0 
    alpha = 0.9 
    qim = QIM(delta)
    k = 0 
    # # x = np.random.uniform(-500, 500, l).astype(float) # host sample
    # x = np.random.randn(l)
    # # x = np.linspace(-5, 5, l).astype(float)  # host sample
    # print('Original x (first 5):', x[:5])

    # msg = qim.random_msg(l)
    # print('Watermark Message (first 10):', msg[:10])
    initial_x = 0.7
    initial_y = 0.
    iterations = 1000
    henon_points = henon_map(initial_x, initial_y, n=iterations)
    alpha = quanti(alpha,delta/10)
    alpha = quasi_periodic(alpha)
    k = quanti(k,delta/10)
    k = quasi_periodic(k)*2/5-0.2
    # print(alpha,k)
    henon_points = np.array(henon_map(alpha,k,n=iterations))
    scale,k = henon_points[:,0]/6+0.75,henon_points[:,1]



    # print(f"\n--- Embedding watermark with fixed alpha = {embedding_alpha}, delta = {delta}, k = {true_k} ---")
    # print(f"")
    # y_watermarked = qim.embed(x, msg, alpha=embedding_alpha, k=true_k)
    # print('Watermarked y (first 5):', y_watermarked[:5])
    # print('Distortion: ',y_watermarked[:5]-x[:5])
    # initial_distortion = np.mean(np.abs(x - y_watermarked))
    
    scale = scale.reshape(-1, 1)
    k_input = k.reshape(-1,1)
    s_mean, s_std = scale.mean(), scale.std()
    scale = (scale - scale.mean()) / scale.std()
    # k_mean, k_std = k_input.mean(), k_input.std()
    # k_input = (k_input - k_input.mean()) / k_input.std()

    # # sys = system_identification(None, henon_points, SS_fix_order=2)  # 'n' is the system order (adjust as needed)
    # basis_function = Legendre(degree=2)
    # model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    # model.fit(y=scale,X=k_input)
    # # y_hat, _, _ = sys.simulate(None, henon_points.shape[0])
    # y_pred = model.predict(y=scale,X=k_input)
    # res = scale - y_pred
    # # mse = np.mean(res**2)
    # fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(scale - scale.mean()))
    # print("\nSystem Identification Results for all at once:")
    # # print(f"MSE: {mse}")
    # print("Fit%:", fit)
    basis_function = Legendre(degree=2)
    model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    y_train, y_valid = scale[:800], scale[800:]
    # 
    x = np.ones(scale.shape)*initial_x
    x_train, x_valid = x[:800], x[800:]

    model.fit(y=y_train, X=x_train)
    y_pred = model.predict(y=y_valid, X=x_valid)

    res = y_valid - y_pred
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(y_valid - y_valid.mean()))
    print("Alpha identification Fit% (validation):", fit)

    basis_function = Legendre(degree=2)
    model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    y = np.ones(scale.shape)*initial_x
    y_train, y_valid = y[:800], y[800:]
    x_train, x_valid = k_input[:800], k_input[800:]
    model.fit(y=x_train, X=y_train)
    x_pred = model.predict(y=x_valid, X=y_valid)

    res = x_valid - x_pred
    # print(x_valid[:10],x_valid.mean())
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(x_valid - x_valid.mean()))
    print("k identification: Fit% (validation):", fit)
    # # Access the state-space matrices
    # A, B, C, D = model.A, model.B, model.C, model.D
    # print("State-space matrices:")
    # print("A:", A)
    # print("B:", B)
    # print("C:", C)
    # print("D:", D)
    scale = scale.reshape(-1, 1)
    k_input = k.reshape(-1,1)
    # s_mean, s_std = scale.mean(), scale.std()
    # scale = (scale - scale.mean()) / scale.std()
    # k_mean, k_std = k_input.mean(), k_input.std()
    # k_input = (k_input - k_input.mean()) / k_input.std()
    y_train, y_valid = scale[:800], scale[800:]
    x_train, x_valid = k_input[:800], k_input[800:]
    # print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    # sys = system_identification(None, henon_points, SS_fix_order=2)  # 'n' is the system order (adjust as needed)
    basis_function = Legendre(degree=1)
    model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    model.fit(X=x_train, y=y_train)
    yhat = model.predict(X=x_valid, y=y_valid)
    res = scale[800:] - yhat
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(y_valid - y_valid.mean(axis=0)))
    print("\nSystem Identification Results for train & validataion:")
    # print(f"MSE: {mse}")
    print("Fit%:", fit)


###################### logistic map #####################################
    qim.fAlpha = ['quasi_periodic','logistic']
    y = qim.alpha_func(initial_x,l).detach().cpu().numpy().reshape(-1,1)
    # y = []
    # x = initial_x
    # for _ in range(l):
    #     y.append(x)
    #     x = logistic_map(x)
    # return np.clip(points, 0.5+eps, 1-eps)  # Ensure alpha is within [0, 1]
    # y = (np.array(y)*0.5+0.5).reshape(-1,1)
    basis_function = Legendre(degree=2)
    model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    y_train, y_valid = y[:800], y[800:]
    # x_train, x_valid = k_input[:800], k_input[800:]
    x = np.ones(y.shape)*initial_x
    x_train, x_valid = x[:800], x[800:]
    model.fit(y=y_train, X=x_train)
    y_pred = model.predict(y=y_valid, X=x_valid)

    res = y_valid - y_pred
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(y_valid - y_valid.mean()))
    print("Logistic Alpha identification Fit% (validation):", fit)

#################### CTBCS ######################################
    qim.fAlpha = ['quasi_periodic','CTBCS']
    y = qim.alpha_func(initial_x,l).detach().cpu().numpy().reshape(-1,1)
    # y = []
    # x = initial_x
    # for _ in range(l):
    #     y.append(x)
    #     x = logistic_map(x)
    # return np.clip(points, 0.5+eps, 1-eps)  # Ensure alpha is within [0, 1]
    # y = (np.array(y)*0.5+0.5).reshape(-1,1)
    basis_function = Legendre(degree=2)
    model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
    y_train, y_valid = y[:800], y[800:]
    # x_train, x_valid = k_input[:800], k_input[800:]
    x = np.ones(y.shape)*initial_x
    x_train, x_valid = x[:800], x[800:]
    model.fit(y=y_train, X=x_train)
    y_pred = model.predict(y=y_valid, X=x_valid)

    res = y_valid - y_pred
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(y_valid - y_valid.mean()))
    print("CTBCS Alpha identification Fit% (validation):", fit)

    # # Prepare input/output
    # scale = (henon_points[:,0]/6 + 0.75).reshape(-1,1)   # output
    # k_input = henon_points[:,1].reshape(-1,1)             # input

    # # Optional: normalize to avoid overflow
    # scale = (scale - scale.mean()) / scale.std()
    # k_input = (k_input - k_input.mean()) / k_input.std()

    # # Train/Validation split
    # train_size = 800
    # y_train, y_valid = scale[:train_size], scale[train_size:]
    # x_train, x_valid = k_input[:train_size], k_input[train_size:]
    # print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    # # x_train, x_valid = henon_points[:800], henon_points[800:]
    # # --- Build NARX model ---
    # basis_function = Polynomial(degree=2)
    # model = NARX(
    #     base_estimator=FROLS(),
    #     xlag=2,
    #     ylag=2,
    #     basis_function=Polynomial(degree=2),
    #     model_type="NARMAX"
    # )

    # # Fit model on training data
    # model.fit(y=y_train, X=x_train)
    # model.fit(X=x_train)
    # yhat = model.predict(X=x_valid)
    # res = scale[800:] - yhat
    # fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(scale[800:] - scale[800:].mean(axis=0)))
    # print("\nNARX System Identification Results for train & validataion:")
    # # print(f"MSE: {mse}")
    # print("Fit%:", fit)
    # Predict on training & validation sets
    # y_train_pred = model.predict(y=y_train, X=x_train)
    # y_valid_pred = model.predict(y=y_valid, X=x_valid)

    # # Compute Fit% for training
    # res_train = y_train - y_train_pred
    # fit_train = 100 * (1 - np.linalg.norm(res_train) / np.linalg.norm(y_train - y_train.mean()))
    # print("Training Fit%:", fit_train)

    # # Compute Fit% for validation
    # res_valid = y_valid - y_valid_pred
    # fit_valid = 100 * (1 - np.linalg.norm(res_valid) / np.linalg.norm(y_valid - y_valid.mean()))
    # print("Validation Fit%:", fit_valid)
