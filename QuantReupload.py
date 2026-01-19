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
df_raw = pd.read_csv('/Users/milan/Desktop/GHQ/data/loto7hh_4548_k5.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)



def quantum_reuploading_predict(df):
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    
    # Prepare lag features
    for col in cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    # Use a small subset for training speed in the kernel
    df_model = df.dropna().tail(15)
    
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
    
    def cost_function(params, X, y):
        mse = 0
        for i in range(len(X)):
            # x_val is the first value, then the weights
            param_values = [X[i][0]] + list(params)
            
            pub = (qc, observable, param_values)
            job = estimator.run([pub])
            result = job.result()[0]
            prediction = result.data.evs
            mse += (prediction - y[i])**2
            
        return mse / len(X)

    for col in cols:
        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values.reshape(-1, 1)
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y).flatten()
        
        # Initial weights for 3 layers (6 params)
        init_params = np.random.rand(num_layers * 2)
        
        # Optimize using COBYLA
        res = minimize(cost_function, init_params, args=(X_scaled, y_scaled), method='COBYLA', options={'maxiter': 20})
        
        # Predict next
        last_val = np.array([[df[col].iloc[-1]]])
        last_val_scaled = scaler_x.transform(last_val)
        
        final_param_values = [last_val_scaled[0][0]] + list(res.x)
        final_pub = (qc, observable, final_param_values)
        final_job = estimator.run([final_pub])
        final_pred_scaled = final_job.result()[0].data.evs
        
        # Inverse scale
        pred_final = scaler_y.inverse_transform(np.array([[final_pred_scaled]]))
        
        # Bound to reasonable lottery numbers
        predictions[col] = max(1, int(round(pred_final[0][0])))
        
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
3-layer Quantum Data Re-uploading Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     5     9    15    12    22    24    35
"""





"""

Multi-Qubit VQR 
QRC 
QNN 
QCNN 
QKA 
QRNN 
QMTR 
QGBR 
QBR 
QSR 






QCM

QDR 

QELM

QGPR 

QTL






quantile 

VQC

QSVR

Quantum Data Re-uploading Regression

"""



"""
ok for VQC and QSVR and Quantum Data Re-uploading Regression and Multi-Qubit VQR and QRC and QNN and QCNN and QKA and QRNN and QMTR and QGBR and QBR and QSR and QDR and QGPR and QTL and QELM, give next model quantum regression with qiskit
"""