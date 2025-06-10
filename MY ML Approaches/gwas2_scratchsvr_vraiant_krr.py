# %% [code] {"papermill":{"duration":68.967797,"end_time":"2025-05-27T09:21:52.179922","exception":false,"start_time":"2025-05-27T09:20:43.212125","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-06-07T11:47:42.150995Z","iopub.execute_input":"2025-06-07T11:47:42.151756Z","iopub.status.idle":"2025-06-07T11:48:41.266444Z","shell.execute_reply.started":"2025-06-07T11:47:42.151731Z","shell.execute_reply":"2025-06-07T11:48:41.265543Z"}}

import numpy as np
import pandas as pd

# %%
#importing Python Libraries
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import variation
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

# %%
file_path="/kaggle/input/usda-complete-data/FINAL_USDA_GENOTYPE.csv"
phenotype_path="/kaggle/input/usda-complete-data/final_phenotype3.csv"
g_data = pd.read_csv(file_path, index_col=0)
p_data = pd.read_csv(phenotype_path, index_col=0)
print("Loaded genotypic data shape:", g_data.shape)

# %%
snp_df=g_data
ph_df=p_data

# %%
snp_df = snp_df.reset_index()  # Brings the original first column back as a regular column
ph_df = ph_df.reset_index()

# %%
ph_df

# %%
y_19 = ph_df["HEIGHT"]
x = snp_df.drop(columns = ['rs#'])
x
y_19 = y_19.fillna(y_19.mean())
print(y_19)

# %% [markdown]
# ### New Encoding AA,TT : 0
# ###              CC,GG : 2
# ###              Other : 1

# %%
snp_encoding = {
    'A': 0, 'C': 2, 'G': 2, 'T': 0,  # Homozygotes as 0
    'R': 1, 'Y': 1, 'S': 1, 'W': 1, 'K': 1, 'M': 1  # Heterozygotes as 1
}

# Function to encode SNP dataset
def encode_snp(df):
    df = df.fillna(-1)
    encoded_df = df.applymap(lambda x: snp_encoding.get(x, np.nan))  # Use np.nan for missing/unknown values
    return encoded_df

encoded_snp_df = encode_snp(x)

encoded_snp_df = encoded_snp_df.fillna(-1)
total_nan = encoded_snp_df.isna().sum().sum()
print(f"Total NaN values in the DataFrame: {total_nan}")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
import torch
import numpy as np
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)  # or any other seed value



import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
class ImprovedSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads=4):
        super(ImprovedSelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        
        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_k * n_heads)
        self.W_o = nn.Linear(d_k * n_heads, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        if len(X.size()) == 2:
            X = X.unsqueeze(1)
        
        batch_size, seq_length, _ = X.size()
        
        Q = self.W_q(X).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))
        
        output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.W_o(output)
        
        return self.layer_norm(X + output).squeeze(1)

class ImprovedAttentionSVR:
    def __init__(self, d_model, d_k, n_heads=16, kernel='poly'):
        self.attention = ImprovedSelfAttention(d_model, d_k, n_heads).to(device)
        self.svr = SVR(kernel=kernel)
        self.scaler = StandardScaler()
        self.optimizer = optim.RAdam(self.attention.parameters(),lr=0.0001)
        #self.optimizer =  AdaHessian(self.attention.parameters(), lr=0.0003, weight_decay=1e-1)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.fc = nn.Linear(d_model, 1).to(device)

    def fit(self, X, y, epochs=5):
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y.values).view(-1, 1).to(device)
        self.losses = [] 
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            X_attended = self.attention(X_tensor)
            output = self.fc(X_attended)
            loss = self.criterion(output, y_tensor)
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        
        X_attended = self.attention(X_tensor).cpu().detach().numpy()
        # print("Shape of attention-enhanced features:", X_attended.shape)
        # print("First 5 rows and 10 columns of attention-enhanced features:\n", X_attended[:5, :10])

        #X_combined = np.concatenate([X, X_attended], axis=1)
        #X_combined = X_attended
        #X_scaled = self.scaler.fit_transform(X_combined)
        # self.svr.fit(X_attended, y)
        # svr_model = ScratchSVR(epsilon=0.3, lambda_=0.7, m=242)
        model = ScratchSVR_L2(epsilon=0.1, lambda_=0.01) 
        model.fit(X_attended, y)
        self.svr = model  # Save it if you need to use it later
        

    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(device)
        X_attended = self.attention(X_tensor).cpu().detach().numpy()
        #X_combined = np.concatenate([X, X_attended], axis=1)
        #X_combined = X_attended  # Replaces original features with attention-enhanced features

        #X_scaled = self.scaler.transform(X_)
        return self.svr.predict(X_attended)




# %% [code]
import numpy as np
import cvxpy as cp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class ScratchSVR_L2:
    def __init__(self, epsilon=0.1, lambda_=0.1):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        # Optimization variables
        w = cp.Variable(n_features)
        b = cp.Variable()
        xi = cp.Variable(n_samples)
        xi_star = cp.Variable(n_samples)

        # Loss: L2 regularization + epsilon-insensitive loss
        objective = cp.Minimize(
            0.5 * cp.sum_squares(w) + self.lambda_ * cp.sum(xi + xi_star)
        )

        # Constraints
        f_x = X @ w + b
        constraints = [
            f_x - y <= self.epsilon + xi,
            y - f_x <= self.epsilon + xi_star,
            xi >= 0,
            xi_star >= 0
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, max_iters=1000, verbose=False)
            self.w = w.value
            self.b = b.value
        except Exception as e:
            print(f"Solver failed: {e}")
            self.w = np.zeros(n_features)
            self.b = 0.0

    def predict(self, X):
        X = np.array(X)
        return (X @ self.w + self.b).flatten()

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred)
        }


# %% [code]
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import numpy as np


def print_summary(name, train, test):
    print(f"\n==={trait_name}: {name} Cross-Validated Performance ===")
    print(f"{name} Train PCC : {np.mean(train['pcc']):.3f} ± {np.std(train['pcc']):.3f}")
    print(f"{name} Train  R² : {np.mean(train['r2']):.3f} ± {np.std(train['r2']):.3f}")
    print(f"{name} Train MSE : {np.mean(train['mse']):.3f} ± {np.std(train['mse']):.3f}")
    print(f"{name} Train MAE : {np.mean(train['mae']):.3f} ± {np.std(train['mae']):.3f}")
    print(f"{name} Test  PCC : {np.mean(test['pcc']):.3f} ± {np.std(test['pcc']):.3f}")
    print(f"{name} Test   R² : {np.mean(test['r2']):.3f} ± {np.std(test['r2']):.3f}")
    print(f"{name} Test  MSE : {np.mean(test['mse']):.3f} ± {np.std(test['mse']):.3f}")
    print(f"{name} Test  MAE : {np.mean(test['mae']):.3f} ± {np.std(test['mae']):.3f}")



r2_scores_svr, mse_scores_svr, mae_scores_svr, pcc_scores_svr = [], [], [], []

trait_list = ["HEIGHT","PROTEIN","OIL","YIELD"]  # Add all trait column names here

for trait_name in trait_list:
    print(f"\n\n===== Starting Cross-Validation for Trait: {trait_name} =====")
    y_trait = ph_df[trait_name].copy()
    y_trait = y_trait.fillna(y_trait.mean())
    X = encoded_snp_df
    # Apply normalization
    # y = normalize_to_minus_one_to_one(y_trait)
    y = y_trait
    kf = KFold(n_splits=5, shuffle=True)

    # Store metrics
    train_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
    test_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

    train_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
    test_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

    r2_scores_svr, mse_scores_svr, mae_scores_svr, pcc_scores_svr = [], [], [], []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
        model = ScratchSVR_L2(epsilon=0.1, lambda_=0.01) 
        # Fit the model
        model.fit(X_train, y_train)
        # Predict on test data
        y_pred = model.predict(X_test)
        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = pearsonr(y_test, y_pred)[0]
        
        # Print metrics
        print("Test Results:")
        print(f"PCC : {pcc:.4f}")
        print(f"R²  : {r2:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"MAE : {mae:.4f}")
        # Store metrics
        r2_scores_svr.append(r2)
        mse_scores_svr.append(mse)
        mae_scores_svr.append(mae)
        pcc_scores_svr.append(pcc)
        # Optional print per fold
        print(f"Variant SVR Fold {fold} — PCC: {pcc:.4f}, R²: {r2:.4f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

        model = ImprovedAttentionSVR(d_model=X_train.shape[1], d_k=128)
        model.fit(X_train.values, y_train)
        y_pred_train_asvr = model.predict(X_train.values)
        y_pred_test_asvr = model.predict(X_test.values)

        train_metrics_asvr['r2'].append(r2_score(y_train, y_pred_train_asvr))
        train_metrics_asvr['mse'].append(mean_squared_error(y_train, y_pred_train_asvr))
        train_metrics_asvr['mae'].append(mean_absolute_error(y_train, y_pred_train_asvr))
        train_metrics_asvr['pcc'].append(pearsonr(y_train, y_pred_train_asvr)[0])

        test_metrics_asvr['r2'].append(r2_score(y_test, y_pred_test_asvr))
        test_metrics_asvr['mse'].append(mean_squared_error(y_test, y_pred_test_asvr))
        test_metrics_asvr['mae'].append(mean_absolute_error(y_test, y_pred_test_asvr))
        test_metrics_asvr['pcc'].append(pearsonr(y_test, y_pred_test_asvr)[0])

        print(f"Variant RAGS: Fold {fold:2d} — Train PCC: {train_metrics_asvr['pcc'][-1]:.3f} | Test PCC: {test_metrics_asvr['pcc'][-1]:.3f}")
        # print(f"Mean PCC: {np.mean(pcc_scores):.4f}")

    print("\nVariant SVR Average Metrics over 10 folds:")
    print(f"Average PCC : {np.mean(pcc_scores_svr):.4f}")
    print(f"Average R²  : {np.mean(r2_scores_svr):.4f}")
    print(f"Average MSE : {np.mean(mse_scores_svr):.2f}")
    print(f"Average MAE : {np.mean(mae_scores_svr):.2f}")

    print(f"RAGS Average PCC over 10 folds: {np.mean(test_metrics_asvr['pcc']):.4f}")
    # print_summary("SVR", train_metrics, test_metrics)
    print_summary("RAGS", train_metrics_asvr, test_metrics_asvr)



# %% [code] {"execution":{"iopub.status.busy":"2025-06-07T11:48:41.267660Z","iopub.execute_input":"2025-06-07T11:48:41.267934Z","iopub.status.idle":"2025-06-07T11:48:41.272219Z","shell.execute_reply.started":"2025-06-07T11:48:41.267915Z","shell.execute_reply":"2025-06-07T11:48:41.271471Z"}}

#Nystroe Feature Reduction Below
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import numpy as np


def print_summary(name, train, test):
    print(f"\n==={trait_name}: {name} Cross-Validated Performance ===")
    print(f"{name} Train PCC : {np.mean(train['pcc']):.3f} ± {np.std(train['pcc']):.3f}")
    print(f"{name} Train  R² : {np.mean(train['r2']):.3f} ± {np.std(train['r2']):.3f}")
    print(f"{name} Train MSE : {np.mean(train['mse']):.3f} ± {np.std(train['mse']):.3f}")
    print(f"{name} Train MAE : {np.mean(train['mae']):.3f} ± {np.std(train['mae']):.3f}")
    print(f"{name} Test  PCC : {np.mean(test['pcc']):.3f} ± {np.std(test['pcc']):.3f}")
    print(f"{name} Test   R² : {np.mean(test['r2']):.3f} ± {np.std(test['r2']):.3f}")
    print(f"{name} Test  MSE : {np.mean(test['mse']):.3f} ± {np.std(test['mse']):.3f}")
    print(f"{name} Test  MAE : {np.mean(test['mae']):.3f} ± {np.std(test['mae']):.3f}")



r2_scores_svr, mse_scores_svr, mae_scores_svr, pcc_scores_svr = [], [], [], []

trait_list = ["HEIGHT","PROTEIN","OIL","YIELD"]  # Add all trait column names here

for trait_name in trait_list:
    print(f"\n\n===== Starting Cross-Validation for Trait: {trait_name} =====")
    y_trait = ph_df[trait_name].copy()
    y_trait = y_trait.fillna(y_trait.mean())
    X = encoded_snp_df
    # Apply normalization
    # y = normalize_to_minus_one_to_one(y_trait)
    y = y_trait
    kf = KFold(n_splits=10, shuffle=True)

    # Store metrics
    train_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
    test_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

    train_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
    test_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

    r2_scores_svr, mse_scores_svr, mae_scores_svr, pcc_scores_svr = [], [], [], []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
        # Step 3: Define kernel-approximated SVR pipeline
        pipeline = make_pipeline(
            StandardScaler(),
            Nystroem(kernel='poly', n_components=300, gamma=1e-4, degree=3, coef0=1),
            SVR(kernel='linear', C=1.0, epsilon=0.1)
        )
    
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(y_pred)
        print(y_test)
        
        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = pearsonr(y_test, y_pred)[0]
        # Store metrics
        r2_scores_svr.append(r2)
        mse_scores_svr.append(mse)
        mae_scores_svr.append(mae)
        pcc_scores_svr.append(pcc)
        # Optional print per fold
        print(f"Variant SVR Fold {fold} — PCC: {pcc:.4f}, R²: {r2:.4f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

        model = ImprovedAttentionSVR(d_model=X_train.shape[1], d_k=128)
        model.fit(X_train.values, y_train)
        y_pred_train_asvr = model.predict(X_train.values)
        y_pred_test_asvr = model.predict(X_test.values)

        train_metrics_asvr['r2'].append(r2_score(y_train, y_pred_train_asvr))
        train_metrics_asvr['mse'].append(mean_squared_error(y_train, y_pred_train_asvr))
        train_metrics_asvr['mae'].append(mean_absolute_error(y_train, y_pred_train_asvr))
        train_metrics_asvr['pcc'].append(pearsonr(y_train, y_pred_train_asvr)[0])

        test_metrics_asvr['r2'].append(r2_score(y_test, y_pred_test_asvr))
        test_metrics_asvr['mse'].append(mean_squared_error(y_test, y_pred_test_asvr))
        test_metrics_asvr['mae'].append(mean_absolute_error(y_test, y_pred_test_asvr))
        test_metrics_asvr['pcc'].append(pearsonr(y_test, y_pred_test_asvr)[0])

        print(f"Variant RAGS: Fold {fold:2d} — Train PCC: {train_metrics_asvr['pcc'][-1]:.3f} | Test PCC: {test_metrics_asvr['pcc'][-1]:.3f}")
        # print(f"Mean PCC: {np.mean(pcc_scores):.4f}")

    print("\nVariant SVR Average Metrics over 10 folds:")
    print(f"Average PCC : {np.mean(pcc_scores_svr):.4f}")
    print(f"Average R²  : {np.mean(r2_scores_svr):.4f}")
    print(f"Average MSE : {np.mean(mse_scores_svr):.2f}")
    print(f"Average MAE : {np.mean(mae_scores_svr):.2f}")

    print(f"RAGS Average PCC over 10 folds: {np.mean(test_metrics_asvr['pcc']):.4f}")
    # print_summary("SVR", train_metrics, test_metrics)
    print_summary("RAGS", train_metrics_asvr, test_metrics_asvr)



# %% [code] {"execution":{"iopub.status.busy":"2025-06-07T11:48:41.289887Z","iopub.execute_input":"2025-06-07T11:48:41.290105Z","iopub.status.idle":"2025-06-07T11:49:27.135042Z","shell.execute_reply.started":"2025-06-07T11:48:41.290080Z","shell.execute_reply":"2025-06-07T11:49:27.134176Z"}}
##KRR Code below

# from sklearn.svm import SVR
# from sklearn.kernel_approximation import Nystroem
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import KFold
# from scipy.stats import pearsonr
# import numpy as np
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.kernel_ridge import KernelRidge


# def print_summary(name, train, test):
#     print(f"\n==={trait_name}: {name} Cross-Validated Performance ===")
#     print(f"{name} Train PCC : {np.mean(train['pcc']):.3f} ± {np.std(train['pcc']):.3f}")
#     print(f"{name} Train  R² : {np.mean(train['r2']):.3f} ± {np.std(train['r2']):.3f}")
#     print(f"{name} Train MSE : {np.mean(train['mse']):.3f} ± {np.std(train['mse']):.3f}")
#     print(f"{name} Train MAE : {np.mean(train['mae']):.3f} ± {np.std(train['mae']):.3f}")
#     print(f"{name} Test  PCC : {np.mean(test['pcc']):.3f} ± {np.std(test['pcc']):.3f}")
#     print(f"{name} Test   R² : {np.mean(test['r2']):.3f} ± {np.std(test['r2']):.3f}")
#     print(f"{name} Test  MSE : {np.mean(test['mse']):.3f} ± {np.std(test['mse']):.3f}")
#     print(f"{name} Test  MAE : {np.mean(test['mae']):.3f} ± {np.std(test['mae']):.3f}")



# trait_list = ["HEIGHT","OIL","PROTEIN","YIELD"]  # Add all trait column names here

# for trait_name in trait_list:
#     print(f"\n\n===== Starting Cross-Validation for Trait: {trait_name} =====")

#     y_trait = ph_df[trait_name].copy()
#     y_trait = y_trait.fillna(y_trait.mean())

#     X = encoded_snp_df
#     # Apply normalization
#     # y = normalize_to_minus_one_to_one(y_trait)
#     y = y_trait
#     kf = KFold(n_splits=10, shuffle=True)

#     # Store metrics
#     train_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
#     test_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

#     train_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
#     test_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}


#     r2_scores_svr, mse_scores_svr, mae_scores_svr, pcc_scores_svr = [], [], [], []
    
#     for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#         # # Step 1: ElasticNet for feature selection (per fold)
#         # enet = ElasticNet(alpha=0.001, l1_ratio=0.5)
#         # enet.fit(X_train, y_train)
#         # important_features = enet.coef_ != 0
    
#         # # Step 2: Reduce features based on selected indices
#         # X_train_selected = X_train.iloc[:, important_features]
#         # X_test_selected = X_test.iloc[:, important_features]
    
#         # print(f"Fold {fold} - X_train_selected shape: {X_train_selected.shape}, X_test_selected shape: {X_test_selected.shape}")
    
#         # Step 3: Define kernel-approximated SVR pipeline
#         pipeline = make_pipeline(
#             StandardScaler(),
#             KernelRidge(kernel='poly', degree=3, coef0=1, alpha=1.0)
#         )
        
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         print(y_pred)
#         print(y_test)
        
#         # Compute metrics
#         r2 = r2_score(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         pcc = pearsonr(y_test, y_pred)[0]
#         # Store metrics
#         r2_scores_svr.append(r2)
#         mse_scores_svr.append(mse)
#         mae_scores_svr.append(mae)
#         pcc_scores_svr.append(pcc)
#         # Optional print per fold
#         print(f"KRR Fold {fold} — PCC: {pcc:.4f}, R²: {r2:.4f}, MSE: {mse:.2f}, MAE: {mae:.2f}")



#         #Model 2 Att+KRR
#         model = ImprovedAttentionSVR(d_model=X_train.shape[1], d_k=128)
#         model.fit(X_train.values, y_train)
#         y_pred_train_asvr = model.predict(X_train.values)
#         y_pred_test_asvr = model.predict(X_test.values)
#         print(y_pred_test_asvr)
#         print(y_test)

#         train_metrics_asvr['r2'].append(r2_score(y_train, y_pred_train_asvr))
#         train_metrics_asvr['mse'].append(mean_squared_error(y_train, y_pred_train_asvr))
#         train_metrics_asvr['mae'].append(mean_absolute_error(y_train, y_pred_train_asvr))
#         train_metrics_asvr['pcc'].append(pearsonr(y_train, y_pred_train_asvr)[0])

#         test_metrics_asvr['r2'].append(r2_score(y_test, y_pred_test_asvr))
#         test_metrics_asvr['mse'].append(mean_squared_error(y_test, y_pred_test_asvr))
#         test_metrics_asvr['mae'].append(mean_absolute_error(y_test, y_pred_test_asvr))
#         test_metrics_asvr['pcc'].append(pearsonr(y_test, y_pred_test_asvr)[0])

#         print(f"ATT+KRR Fold {fold:2d} — Train PCC: {train_metrics_asvr['pcc'][-1]:.3f} | Test PCC: {test_metrics_asvr['pcc'][-1]:.3f}")
#         # print(f"Mean PCC: {np.mean(pcc_scores):.4f}")
#         break

#     print("\nAverage (Kernel Ridge) Average Metrics over 10 folds:")
#     print(f"Average PCC : {np.mean(pcc_scores_svr):.4f}")
#     print(f"Average R²  : {np.mean(r2_scores_svr):.4f}")
#     print(f"Average MSE : {np.mean(mse_scores_svr):.2f}")
#     print(f"Average MAE : {np.mean(mae_scores_svr):.2f}")

#     print(f"RAGS Average PCC over 10 folds: {np.mean(test_metrics_asvr['pcc']):.4f}")
#     # print_summary("SVR", train_metrics, test_metrics)
#     print_summary("RAGS", train_metrics_asvr, test_metrics_asvr)



# # print_summary("SVR", train_metrics, test_metrics)
# # print_summary("RAGS", train_metrics_asvr, test_metrics_asvr)


# %% [code] {"execution":{"iopub.status.busy":"2025-06-02T13:52:15.397070Z","iopub.execute_input":"2025-06-02T13:52:15.397469Z","iopub.status.idle":"2025-06-02T14:11:49.118882Z","shell.execute_reply.started":"2025-06-02T13:52:15.397450Z","shell.execute_reply":"2025-06-02T14:11:49.117917Z"}}
# from sklearn.model_selection import KFold
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from scipy.stats import pearsonr
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from sklearn.linear_model import ElasticNet


# # def normalize_to_minus_one_to_one(y):
# #     y_min = y.min()
# #     y_max = y.max()
# #     return 2 * (y - y_min) / (y_max - y_min) - 1
# # from bayes_opt import BayesianOptimization
# # from sklearn.model_selection import train_test_split
# # from scipy.stats import pearsonr

# # # Define objective function for Bayesian Optimization
# # def svr_pcc_objective(epsilon, lambda_, alpha):
# #     X_sub, y_sub = encoded_snp_df.values, ph_df["GY_2019"].fillna(ph_df["GY_2019"].mean()).values
# #     X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.1)

# #     svr = ScratchSVR(epsilon=epsilon, lambda_=lambda_, m=X_train.shape[0])
# #     svr.alpha = alpha  # Custom addition: make alpha part of the object (or pass separately)
# #     svr.fit(X_train, y_train)
# #     y_pred = svr.predict(X_val)

# #     pcc, _ = pearsonr(y_val, y_pred)
# #     return pcc

# # # Define the hyperparameter search space
# # pbounds = {
# #     'epsilon': (0.05, 0.25),   # Narrow range for epsilon
# #     'lambda_': (0.01, 0.15),   # Smaller regularization range
# #     'alpha': (0.01, 1.0)      # Focus on smaller alpha values
# # }


# # optimizer = BayesianOptimization(
# #     f=svr_pcc_objective,
# #     pbounds=pbounds,
# #     verbose=2,
# #     random_state=None  # let randomness vary
# # )

# # optimizer.maximize(init_points=3, n_iter=10)  # total 8 iterations


# # # Get best parameters
# # best_params = optimizer.max['params']
# # print("Best ScratchSVR Parameters:", best_params)



# trait_list = ["PH_2019","NN_2019","GY_2019","CT_2019"]  # Add all trait column names here
# #normalize to [-1 to 1]
# for trait_name in trait_list:
#     print(f"\n\n===== Starting Cross-Validation for Trait: {trait_name} =====")

#     y_trait = ph_df[trait_name].copy()
#     y_trait = y_trait.fillna(y_trait.mean())

#     X = encoded_snp_df
#     # Apply normalization
#     # y = normalize_to_minus_one_to_one(y_trait)
#     y = y_trait
#     kf = KFold(n_splits=10, shuffle=True)

#     # Store metrics
#     train_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
#     test_metrics = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

#     train_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}
#     test_metrics_asvr = {'pcc': [], 'r2': [], 'mse': [], 'mae': []}

#     for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#         # 2) ElasticNet for feature selection
#         enet = ElasticNet(alpha=0.001, l1_ratio=0.5)
#         enet.fit(X_train, y_train)
        
#         important_features = enet.coef_ != 0
        
#         # Reduce features
#         X_train_selected = X_train.iloc[:, important_features]
#         X_test_selected = X_test.iloc[:, important_features]
    
#         print("X_train_selected shape:", X_train_selected.shape)
#         print("X_test_selected shape:", X_test_selected.shape)

        
#         # # SVR baseline
#         # svr = SVR(kernel='poly')
#         # svr.fit(X_train_selected, y_train)
#         # y_pred_train_svr = svr.predict(X_train_selected)
#         # y_pred_test_svr = svr.predict(X_test_selected)

#         #Calling Our Scratch SVR here
        
#         svr = ScratchSVR(epsilon=0.3, lambda_=0.7, m=242)
        
#         # svr = ScratchSVR(
#         # epsilon=best_params['epsilon'],
#         # lambda_=best_params['lambda_'],
#         # m=X_train_selected.shape[0]
#         # )
        
#         # svr.alpha = best_params['alpha']
#         svr.fit(X_train_selected, y_train)
#         y_pred_train_svr = svr.predict(X_train_selected)
#         y_pred_test_svr = svr.predict(X_test_selected)


#         # Train
#         train_metrics['r2'].append(r2_score(y_train, y_pred_train_svr))
#         train_metrics['mse'].append(mean_squared_error(y_train, y_pred_train_svr))
#         train_metrics['mae'].append(mean_absolute_error(y_train, y_pred_train_svr))
#         train_metrics['pcc'].append(pearsonr(y_train, y_pred_train_svr)[0])

#         # Test
#         test_metrics['r2'].append(r2_score(y_test, y_pred_test_svr))
#         test_metrics['mse'].append(mean_squared_error(y_test, y_pred_test_svr))
#         test_metrics['mae'].append(mean_absolute_error(y_test, y_pred_test_svr))
#         test_metrics['pcc'].append(pearsonr(y_test, y_pred_test_svr)[0])

#         print(f"Normal SVR:  Fold {fold:2d} — Train PCC: {train_metrics['pcc'][-1]:.3f} | Test PCC: {test_metrics['pcc'][-1]:.3f}")

#         # Attention + SVR model
#         model = ImprovedAttentionSVR(d_model=X_train_selected.shape[1], d_k=128)
#         model.fit(X_train_selected.values, y_train)
#         y_pred_train_asvr = model.predict(X_train_selected.values)
#         y_pred_test_asvr = model.predict(X_test_selected.values)

#         train_metrics_asvr['r2'].append(r2_score(y_train, y_pred_train_asvr))
#         train_metrics_asvr['mse'].append(mean_squared_error(y_train, y_pred_train_asvr))
#         train_metrics_asvr['mae'].append(mean_absolute_error(y_train, y_pred_train_asvr))
#         train_metrics_asvr['pcc'].append(pearsonr(y_train, y_pred_train_asvr)[0])

#         test_metrics_asvr['r2'].append(r2_score(y_test, y_pred_test_asvr))
#         test_metrics_asvr['mse'].append(mean_squared_error(y_test, y_pred_test_asvr))
#         test_metrics_asvr['mae'].append(mean_absolute_error(y_test, y_pred_test_asvr))
#         test_metrics_asvr['pcc'].append(pearsonr(y_test, y_pred_test_asvr)[0])

#         print(f"Normal RAGS: Fold {fold:2d} — Train PCC: {train_metrics_asvr['pcc'][-1]:.3f} | Test PCC: {test_metrics_asvr['pcc'][-1]:.3f}")

#     # === Summary ===
#     def print_summary(name, train, test):
#         print(f"\n==={trait_name}: {name} Cross-Validated Performance ===")
#         print(f"{name} Train PCC : {np.mean(train['pcc']):.3f} ± {np.std(train['pcc']):.3f}")
#         print(f"{name} Train  R² : {np.mean(train['r2']):.3f} ± {np.std(train['r2']):.3f}")
#         print(f"{name} Train MSE : {np.mean(train['mse']):.3f} ± {np.std(train['mse']):.3f}")
#         print(f"{name} Train MAE : {np.mean(train['mae']):.3f} ± {np.std(train['mae']):.3f}")

#         print(f"{name} Test  PCC : {np.mean(test['pcc']):.3f} ± {np.std(test['pcc']):.3f}")
#         print(f"{name} Test   R² : {np.mean(test['r2']):.3f} ± {np.std(test['r2']):.3f}")
#         print(f"{name} Test  MSE : {np.mean(test['mse']):.3f} ± {np.std(test['mse']):.3f}")
#         print(f"{name} Test  MAE : {np.mean(test['mae']):.3f} ± {np.std(test['mae']):.3f}")

#     print_summary("SVR", train_metrics, test_metrics)
#     print_summary("RAGS", train_metrics_asvr, test_metrics_asvr)

#     # === Plotting ===
#     def plot_metrics(metric_dict_train, metric_dict_test, model_name):
#         fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#         metrics = ['pcc', 'r2', 'mse', 'mae']
#         titles = ['PCC', 'R²', 'MSE', 'MAE']

#         for i, (metric, title) in enumerate(zip(metrics, titles)):
#             row, col = divmod(i, 2)
#             axs[row][col].plot(metric_dict_train[metric], label='Train', marker='o')
#             axs[row][col].plot(metric_dict_test[metric], label='Test', marker='x')
#             axs[row][col].set_title(title)
#             axs[row][col].set_xlabel('Fold')
#             axs[row][col].set_ylabel(title)
#             axs[row][col].legend()
#             axs[row][col].grid(True)

#         plt.suptitle(f'{model_name} Performance on Trait: {trait_name}', fontsize=14)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#         os.makedirs("plots", exist_ok=True)
#         plt.savefig(f"plots/{trait_name}_{model_name}.png")
#         plt.close()

#     plot_metrics(train_metrics, test_metrics, "SVR")
#     plot_metrics(train_metrics_asvr, test_metrics_asvr, "RAGS")

# %% [markdown]
# # **Byesian Optimization on Elastic Net# **

# %% [code] {"execution":{"iopub.status.busy":"2025-05-28T10:10:30.292820Z","iopub.execute_input":"2025-05-28T10:10:30.293397Z","iopub.status.idle":"2025-05-28T10:33:28.134129Z","shell.execute_reply.started":"2025-05-28T10:10:30.293377Z","shell.execute_reply":"2025-05-28T10:33:28.133159Z"}}
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.linear_model import ElasticNet
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
# from scipy.stats import pearsonr
# from skopt import BayesSearchCV

# # Custom PCC scorer function
# def pcc_scorer(y_true, y_pred):
#     return pearsonr(y_true, y_pred)[0]

# pcc_score = make_scorer(pcc_scorer, greater_is_better=True)

# r2_scores, rmse_scores, mae_scores, pcc_scores = [], [], [], []

# kf = KFold(n_splits=10, shuffle=True)  # shuffle with varying seeds each run


# X = encoded_snp_df
# y = y_19
# print(y_19)

# for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
#     # ElasticNet hyperparameter search with Bayesian Optimization maximizing PCC
#     param_grid = {
#         'alpha': (1e-4, 1.0, 'log-uniform'),
#         'l1_ratio': (0.1, 0.9, 'uniform')
#     }
    
#     enet = ElasticNet(max_iter=3000)
#     opt = BayesSearchCV(
#         enet,
#         param_grid,
#         n_iter=5,  # increase iterations for better optimization
#         cv=2,
#         scoring=pcc_score,  # optimize PCC
#         n_jobs=-1,
#         verbose=0,
#         random_state=None
#     )
#     opt.fit(X_train, y_train)

#     best_enet = opt.best_estimator_
#     important_features = best_enet.coef_ != 0

#     X_train_selected = X_train.loc[:, important_features]
#     X_test_selected = X_test.loc[:, important_features]

#     # SVR with linear kernel (you can experiment here too)
#     svr = SVR(kernel='poly')
#     svr.fit(X_train_selected, y_train)
#     y_pred = svr.predict(X_test_selected)

#     # Metrics
#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     pcc = pearsonr(y_test, y_pred)[0]

#     r2_scores.append(r2)
#     rmse_scores.append(rmse)
#     mae_scores.append(mae)
#     pcc_scores.append(pcc)

#     print(f"Fold {fold:02d}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, PCC={pcc:.4f}")

# print("\nAverage Performance Across 10 Folds:")
# print(f"R2:  {np.mean(r2_scores):.4f}")
# print(f"RMSE:{np.mean(rmse_scores):.4f}")
# print(f"MAE: {np.mean(mae_scores):.4f}")
# print(f"PCC: {np.mean(pcc_scores):.4f}")


# %% [code] {"execution":{"iopub.status.busy":"2025-05-28T11:22:36.546357Z","iopub.execute_input":"2025-05-28T11:22:36.546648Z","iopub.status.idle":"2025-05-28T11:24:48.378186Z","shell.execute_reply.started":"2025-05-28T11:22:36.546630Z","shell.execute_reply":"2025-05-28T11:24:48.377321Z"},"papermill":{"duration":313.985226,"end_time":"2025-05-27T09:27:06.166929","exception":false,"start_time":"2025-05-27T09:21:52.181703","status":"completed"},"tags":[]}
# # %%
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import pearsonr

# X = encoded_snp_df
# y = y_19

# k = 10
# kf = KFold(n_splits=k, shuffle=True)

# r2_scores_svr  = []
# mse_scores_svr = []
# mae_scores_svr = []
# pcc_scores_svr = []

# r2_scores_asvr  = []
# mse_scores_asvr = []
# mae_scores_asvr = []
# pcc_scores_asvr = []


# coef_list = []
# coef_list_asvr = []

# for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
#     # 1) Split
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#     # 2) ElasticNet for feature selection
#     enet = ElasticNet(alpha=0.001, l1_ratio=0.5)
#     enet.fit(X_train, y_train)
#     important_features = enet.coef_ != 0
#     # Reduce features
#     X_train_selected = X_train.iloc[:, important_features]
#     X_test_selected = X_test.iloc[:, important_features]
#     print("X_train_selected shape:", X_train_selected.shape)
#     print("X_test_selected shape:", X_test_selected.shape)
#     svr = SVR(kernel='poly')
#     svr.fit(X_train_selected, y_train)
#     y_pred_svr = svr.predict(X_test_selected)
#     r2_svr  = r2_score(y_test, y_pred_svr)
#     mse_svr = mean_squared_error(y_test, y_pred_svr)
#     mae_svr = mean_absolute_error(y_test, y_pred_svr)
#     pcc_svr,_ = pearsonr(y_test, y_pred_svr)
    

#     r2_scores_svr.append(r2_svr)
#     mse_scores_svr.append(mse_svr)
#     mae_scores_svr.append(mae_svr)
#     pcc_scores_svr.append(pcc_svr)
#     print(f"SVR:  Fold {fold:2d} — PCC: {pcc_svr:.3f} | R2: {r2_svr:.3f} │ MSE: {mse_svr:.3f} │ MAE: {mae_svr:.3f}")
#     coef = np.dot(svr.dual_coef_, svr.support_vectors_).flatten()
#     coef_list.append(coef)


    
#     model = ImprovedAttentionSVR(d_model=X_train_selected.shape[1], d_k=64)
#     model.fit(X_train_selected.values, y_train)
#     y_pred_asvr = model.predict(X_test_selected.values)
#     r2_asvr  = r2_score(y_test, y_pred_asvr)
#     mse_asvr = mean_squared_error(y_test, y_pred_asvr)
#     mae_asvr = mean_absolute_error(y_test, y_pred_asvr)
#     pcc_asvr,_ = pearsonr(y_test, y_pred_asvr)
#     pcc_scores_asvr.append(pcc_asvr)
#     r2_scores_asvr.append(r2_asvr)
#     mse_scores_asvr.append(mse_asvr)
#     mae_scores_asvr.append(mae_asvr)
#     print(f"RAGS: Fold {fold:2d} — PCC: {pcc_asvr:.3f} | R2: {r2_asvr:.3f} │ MSE: {mse_asvr:.3f} │ MAE: {mae_asvr:.3f}")
#     coef_asvr = np.dot(model.svr.dual_coef_, model.svr.support_vectors_).flatten()
#     coef_list_asvr.append(coef_asvr)


# #TOP 20 SNP SVR
# avg_abs_coef = np.mean(np.abs(coef_list), axis=0)
# # Select top 20 SNPs
# sorted_indices = np.argsort(avg_abs_coef)[::-1]
# top_20_indices = sorted_indices[:20]
# # Map indices to SNP names
# feature_names = list(encoded_snp_df.columns)
# top_20_snps = [feature_names[idx] for idx in top_20_indices]
# top_20_importance = avg_abs_coef[top_20_indices]
# # Save to CSV
# results_df = pd.DataFrame({'SNP': top_20_snps, 'Importance': top_20_importance})
# results_df.to_csv("OIL_SVR_ALL_Best_SNPs.csv", index=False)
# print("Saved top 20 averaged SNPs to 'OIL_SVR_ALL_Best_SNPs.csv'.")


# #TOP 20 SNP ASVR
# avg_abs_coef_2 = np.mean(np.abs(coef_list_asvr), axis=0)
# # Select top 20 SNPs
# sorted_indices_2 = np.argsort(avg_abs_coef_2)[::-1]
# top_20_indices_2 = sorted_indices_2[:20]
# # Map indices to SNP names
# feature_names_2 = list(encoded_snp_df.columns)
# top_20_snps_2 = [feature_names_2[idx] for idx in top_20_indices_2]
# top_20_importance_2 = avg_abs_coef_2[top_20_indices_2]
# # Save to CSV
# results_df_2 = pd.DataFrame({'SNP': top_20_snps_2, 'Importance': top_20_importance_2})
# results_df_2.to_csv("OIL_ASVR_ALL_Best_SNPs.csv", index=False)
# print("Saved top 20 averaged SNPs to 'OIL_ALL_Best_SNPs.csv'.")


# # 8) Final averages
# print(f"\n==={y_19.name}: SVR Cross-Validated Performance ===")
# print(f"SVR: Mean PCC : {np.mean(pcc_scores_svr)} ± {np.std(pcc_scores_svr)}")
# print(f"SVR: Mean  R² : {np.mean(r2_scores_svr):.3f} ± {np.std(r2_scores_svr):.3f}")
# print(f"SVR: Mean MSE : {np.mean(mse_scores_svr):.3f} ± {np.std(mse_scores_svr):.3f}")
# print(f"SVR: Mean MAE : {np.mean(mae_scores_svr):.3f} ± {np.std(mae_scores_svr):.3f}")

# print(f"\n==={y_19.name}: RAGS Cross-Validated Performance ===")
# print(f"RAGS: Mean PCC : {np.mean(pcc_scores_asvr)} ± {np.std(pcc_scores_asvr)}")
# print(f"RAGS: Mean  R² : {np.mean(r2_scores_asvr):.3f} ± {np.std(r2_scores_asvr):.3f}")
# print(f"RAGS: Mean MSE : {np.mean(mse_scores_asvr):.3f} ± {np.std(mse_scores_asvr):.3f}")
# print(f"RAGS: Mean MAE : {np.mean(mae_scores_asvr):.3f} ± {np.std(mae_scores_asvr):.3f}")

