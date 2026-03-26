# 3-layer Quantum Data Re-uploading for Lottery Prediction
# Quantum Regression Model with Qiskit


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize


from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7hh_4586_k24.csv')
# 4586 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def quantum_reuploading_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    
    # Prepare lag features
    for col in cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    # Use a small subset for training speed in the kernel
    df_model = df.dropna().tail(1200)
    
    predictions = {}
    
    # Scaling
    scaler_x = MinMaxScaler(feature_range=(0, np.pi))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    # Quantum Circuit Setup: Data Re-uploading (3 layers)
    num_qubits = 1
    num_layers = 3
    
    # Define parameters
    x = Parameter('x') # Single input parameter used in every layer
    theta = ParameterVector('theta', num_layers * 2) # 2 trainable weights per layer
    
    qc = QuantumCircuit(num_qubits)
    for i in range(num_layers):
        # Data encoding (Re-uploading)
        qc.ry(x, 0)
        # Trainable ansatz
        qc.rz(theta[i*2], 0)
        qc.ry(theta[i*2 + 1], 0)
    
    observable = SparsePauliOp('Z')
    estimator = StatevectorEstimator()
    
    # The unique parameters in the circuit are [x, theta[0], ..., theta[5]]
    # We must match this order in the binding list
    circuit_params = [x] + list(theta)
    
    def eval_pred(x_val, params):
        param_values = [x_val] + list(params)
        pub = (qc, observable, param_values)
        job = estimator.run([pub])
        result = job.result()[0]
        evs = result.data.evs
        return float(np.real(np.asarray(evs).reshape(-1)[0]))

    def cost_function(params, X, y):
        mse = 0.0
        for i in range(len(X)):
            prediction = eval_pred(X[i][0], params)
            mse += (prediction - y[i])**2
            
        return mse / len(X)

    for idx, col in enumerate(cols):
        print(f"\n[QuantReupload_v2] pozicija {idx+1} ({col})...")
        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values.reshape(-1, 1)
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y).flatten()

        # Brži podskup za optimizaciju (predikcija ostaje iz poslednje vrednosti)
        n_sub = min(100, len(X_scaled))
        X_opt = X_scaled[-n_sub:]
        y_opt = y_scaled[-n_sub:]
        
        # Initial weights for 3 layers (6 params)
        best_x = None
        best_cost = float("inf")
        for restart in range(2):
            init_params = np.random.uniform(0, 2*np.pi, num_layers * 2)
            
            # Optimize using COBYLA
            res = minimize(
                cost_function,
                init_params,
                args=(X_opt, y_opt),
                method='COBYLA',
                options={'maxiter': 60, 'rhobeg': 0.25}
            )
            c = float(res.fun)
            if c < best_cost:
                best_cost = c
                best_x = res.x
            print(f"[QuantReupload_v2] restart {restart+1}/2 cost={c:.6f} best={best_cost:.6f}")
        
        # Predict next
        last_val = np.array([[df[col].iloc[-1]]])
        last_val_scaled = scaler_x.transform(last_val)
        
        final_pred_scaled = eval_pred(last_val_scaled[0][0], best_x)
        
        # Inverse scale
        pred_final = scaler_y.inverse_transform(np.array([[final_pred_scaled]]))
        
        # Bound to reasonable lottery numbers
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(pred_final[0][0], lo, hi)))
        
    return predictions

print()
print("Computing predictions using Quantum Data Re-uploading Regression ...")
print()
q_reupload_results = quantum_reuploading_predict(df_raw)

# Format for display
q_reupload_df = pd.DataFrame([q_reupload_results])
# q_reupload_df.index = ['Quantum Data Re-uploading Prediction']

print()
print("Lottery prediction generated using a 3-layer Quantum Data Re-uploading model.")
print()


print()
print("3-layer Quantum Data Re-uploading Results:")
print(q_reupload_df.to_string(index=True))
print()
"""
Computing predictions using Quantum Data Re-uploading Regression ...


[QuantReupload_v2] pozicija 1 (Num1)...
[QuantReupload_v2] restart 1/2 cost=0.118118 best=0.118118
[QuantReupload_v2] restart 2/2 cost=0.118291 best=0.118118

[QuantReupload_v2] pozicija 2 (Num2)...
[QuantReupload_v2] restart 1/2 cost=0.171723 best=0.171723
[QuantReupload_v2] restart 2/2 cost=0.172632 best=0.171723

[QuantReupload_v2] pozicija 3 (Num3)...
[QuantReupload_v2] restart 1/2 cost=0.193403 best=0.193403
[QuantReupload_v2] restart 2/2 cost=0.199068 best=0.193403

[QuantReupload_v2] pozicija 4 (Num4)...
[QuantReupload_v2] restart 1/2 cost=0.175931 best=0.175931
[QuantReupload_v2] restart 2/2 cost=0.176804 best=0.175931

[QuantReupload_v2] pozicija 5 (Num5)...
[QuantReupload_v2] restart 1/2 cost=0.160077 best=0.160077
[QuantReupload_v2] restart 2/2 cost=0.162397 best=0.160077

[QuantReupload_v2] pozicija 6 (Num6)...
[QuantReupload_v2] restart 1/2 cost=0.166545 best=0.166545
[QuantReupload_v2] restart 2/2 cost=0.147811 best=0.147811

[QuantReupload_v2] pozicija 7 (Num7)...
[QuantReupload_v2] restart 1/2 cost=0.068835 best=0.068835
[QuantReupload_v2] restart 2/2 cost=0.076992 best=0.068835

Lottery prediction generated using a 3-layer Quantum Data Re-uploading model.


3-layer Quantum Data Re-uploading Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     4    10    x    y    z    30    36
"""


"""
df.copy() da se ulaz ne muti,
prozor tail(1200) umesto celog seta,
brži fit na podskupu (last 100 za optimizaciju),
COBYLA ubrzan i stabilniji (2 restarta, maxiter=60, rhobeg=0.25),
sigurno čitanje kao float,
clip po pozicijama za sortirani 7/39 (1..33 … 7..39),
dodati progres printovi po poziciji/restartu da ne deluje da stoji.
"""
