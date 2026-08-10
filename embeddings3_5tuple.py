import pandas as pd
import numpy as np
import gc
import os
import sys
from tqdm import tqdm
from scipy.fft import rfft
from sklearn.linear_model import LinearRegression
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# 1. MOTOR MATEMÁTICO (FEATURE EXTRACTOR)
# ==============================================================================

class TrafficFeatureExtractor:
    def __init__(self):
        pass

    # --- ESTATÍSTICAS BÁSICAS ---
    def compute_stats(self, matrix_X):
        if matrix_X.shape[0] == 0: return np.zeros(15, dtype=np.float64)
        means = np.mean(matrix_X, axis=0)
        medians = np.median(matrix_X, axis=0)
        stds = np.std(matrix_X, axis=0)
        maxs = np.max(matrix_X, axis=0)
        mins = np.min(matrix_X, axis=0)
        return np.concatenate([means, medians, stds, maxs, mins])

    # --- ENERGIA E ENTROPIA ---
    def compute_l2_norm(self, matrix_X):
        if matrix_X.shape[0] == 0: return np.zeros(3, dtype=np.float64)
        return np.sqrt(np.sum(matrix_X**2, axis=0))

    def compute_entropy_shannon(self, matrix_X):
        if matrix_X.shape[0] == 0: return np.zeros(3, dtype=np.float64)
        entropies = []
        for i in range(matrix_X.shape[1]):
            col = np.abs(matrix_X[:, i])
            soma = np.sum(col)
            if soma == 0:
                entropies.append(0.0)
            else:
                probs = col / soma
                probs = probs[probs > 0]
                entr = -np.sum(probs * np.log2(probs))
                entropies.append(entr)
        return np.array(entropies, dtype=np.float64)

    # --- BURSTINESS ---
    def compute_burstiness(self, matrix_X):
        if matrix_X.shape[0] < 2: return np.zeros(3, dtype=np.float64)
        means = np.mean(matrix_X, axis=0)
        vars_ = np.var(matrix_X, axis=0)
        means[means == 0] = 1e-9
        return vars_ / means

    # --- ANÁLISE NO DOMÍNIO DA FREQUÊNCIA (FFT) ---
    def compute_frequency_features(self, matrix_X):
        if matrix_X.shape[0] < 4: return np.zeros(4, dtype=np.float64)
        feats = []
        for col_idx in [0, 2]: # Size e IAT
            data = matrix_X[:, col_idx]
            data_centered = data - np.mean(data)
            fft_vals = np.abs(rfft(data_centered))
            if len(fft_vals) > 1:
                fft_vals = fft_vals[1:] 
                max_power = np.max(fft_vals)
                dom_freq_norm = np.argmax(fft_vals) / len(fft_vals)
            else:
                max_power = 0.0; dom_freq_norm = 0.0
            feats.extend([max_power, dom_freq_norm])
        return np.array(feats, dtype=np.float64)

    # --- ÁLGEBRA LINEAR ---
    def compute_pca_features(self, data, K=3):
        if data.shape[0] < 2: return np.zeros(K)
        try:
            cov = np.cov(data, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            return np.sort(eigvals)[::-1][:K] if len(eigvals) >= K else np.zeros(K)
        except: return np.zeros(K)

    def compute_var_features(self, data, p=1):
        N, d = data.shape 
        if N <= p + 2: return np.zeros(d*d*p)
        try:
            Y_list = [data[:-lag] for lag in range(1, p + 1)]
            Y = np.hstack(Y_list)
            X = data[p:]
            stds = np.std(Y, axis=0)
            stds[stds == 0] = 1.0
            Y_norm = (Y - np.mean(Y, axis=0)) / stds
            A_mat, *_ = np.linalg.lstsq(Y_norm, X, rcond=None)
            return A_mat.flatten()
        except: return np.zeros(d*d*p)

    def generate_embedding(self, matrix_X):
        if matrix_X.shape[0] == 0: return None
        return np.concatenate([
            self.compute_stats(matrix_X),           # 15
            self.compute_l2_norm(matrix_X),         # 3
            self.compute_entropy_shannon(matrix_X), # 3
            self.compute_burstiness(matrix_X),      # 3
            self.compute_frequency_features(matrix_X), # 4
            self.compute_pca_features(matrix_X, K=3), # 3
            self.compute_var_features(matrix_X, p=1)  # 9
        ])

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def get_feature_names():
    cols = ['size', 'time', 'iat']
    names = []
    # Stats (15)
    for m in ['mean', 'median', 'std', 'max', 'min']:
        for c in cols: names.append(f"{m}_{c}")
    # L2, Entropy, Burstiness (3 cada)
    for t in ['l2', 'entropy', 'burstiness']:
        for c in cols: names.append(f"{t}_{c}")
    # FFT (4)
    names.extend(["fft_power_size", "fft_freq_size", "fft_power_iat", "fft_freq_iat"])
    # PCA (3)
    for i in range(1, 4): names.append(f"pca_eig_{i}")
    # VAR (9)
    for i in range(9): names.append(f"var_coeff_{i}")
    return names

def build_invariant_matrix(df_group, size_col, time_col):
    df_sorted = df_group.sort_values(time_col)
    X_vec = df_sorted[size_col].values.astype(np.float64)
    times = df_sorted[time_col].values.astype(np.float64)
    
    if len(times) > 0:
        t0 = times[0]
        iat = np.diff(times, prepend=t0)
        iat[0] = 0.0 
        T_prime = times - t0
    else:
        T_prime = np.array([], dtype=np.float64)
        iat = np.array([], dtype=np.float64)
        
    return np.column_stack((X_vec, T_prime, iat))

def load_data(file_path):
    print(f"--- Lendo arquivo: {file_path} ---")
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)

# ==============================================================================
# 3. PIPELINE DE EXECUÇÃO
# ==============================================================================

def run_pipeline(config):
    input_file = config['IO']['INPUT_FILE']
    output_dir = config['IO']['OUTPUT_DIR']
    output_filename = config['IO']['OUTPUT_FILENAME']
    batch_size = config['HYPERPARAMETERS']['BATCH_SIZE']
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    final_path = os.path.join(output_dir, output_filename)

    # Configurações de Colunas
    MAP = config['COLUMNS']
    df = load_data(input_file)
    
    df['global_index'] = df.index.astype(int)
    if MAP['label_multi'] not in df.columns: df[MAP['label_multi']] = "Unknown"
    if MAP['label_binary'] not in df.columns: df[MAP['label_binary']] = "0"

    key_cols = [MAP['src_ip'], MAP['dst_ip'], MAP['src_port'], MAP['dst_port'], MAP['protocol']]
    
    extractor = TrafficFeatureExtractor()
    feat_names = get_feature_names()
    
    print(f"\n🚀 Iniciando Extração (Strategy: Packets per Flow)")
    print(f"   └── Batch Size: {batch_size}")
    print(f"   └── Filtro: Ignorando fluxos com < 3 pacotes")
    print(f"   └── Arquivo Final Único: {final_path}")
    
    groups = df.groupby(key_cols)
    
    # --- PREPARAÇÃO DO PARQUET WRITER ---
    pq_writer = None
    current_batch = []
    batch_counter = 0
    total_flows = 0
    skipped_flows = 0
    
    for name, group in tqdm(groups, desc="Processando Fluxos"):
        
        # --- FILTRO DE TAMANHO ---
        # Ignora fluxos muito curtos (ruído estatístico)
        if len(group) < 3: 
            skipped_flows += 1
            continue
            
        matrix_X = build_invariant_matrix(group, MAP['size'], MAP['timestamp'])
        emb_vector = extractor.generate_embedding(matrix_X)
        
        if emb_vector is not None:
            emb_dict = dict(zip(feat_names, emb_vector))
            
            # Metadados
            emb_dict['Label'] = str(group[MAP['label_multi']].iloc[0])
            emb_dict['Label_Type'] = str(group[MAP['label_binary']].iloc[0])
            emb_dict['Flow_ID'] = str(name)
            emb_dict['Packet_Count'] = len(group)
            emb_dict['src_packet_indices'] = group['global_index'].tolist()
            
            current_batch.append(emb_dict)
            total_flows += 1
            
            # --- WRITE BATCH ---
            if len(current_batch) >= batch_size:
                df_temp = pd.DataFrame(current_batch)
                table = pa.Table.from_pandas(df_temp)
                
                if pq_writer is None:
                    pq_writer = pq.ParquetWriter(final_path, table.schema)
                
                pq_writer.write_table(table)
                
                batch_counter += 1
                current_batch = []
                del df_temp, table
                gc.collect()
    
    # --- FLUSH FINAL ---
    if len(current_batch) > 0:
        df_temp = pd.DataFrame(current_batch)
        table = pa.Table.from_pandas(df_temp)
        
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(final_path, table.schema)
            
        pq_writer.write_table(table)
        del df_temp, table
        gc.collect()

    # --- FECHAMENTO ---
    if pq_writer:
        pq_writer.close()

    print(f"\n✅ Concluído com Sucesso!")
    print(f"   └── Arquivo Gerado: {final_path}")
    print(f"   └── Fluxos Processados: {total_flows}")
    print(f"   └── Fluxos Ignorados (<3 pacotes): {skipped_flows}")

# ==============================================================================
# 4. CONFIGURAÇÃO
# ==============================================================================

if __name__ == "__main__":
    dataset = "edgeiiot"
    tipo = "all"
    
    PAINEL_CONFIG = {
        'IO': {
            'INPUT_FILE': f"datasets/{dataset}/{tipo}_{dataset}.parquet",
            'OUTPUT_DIR': f"output_{dataset}_final3",
            'OUTPUT_FILENAME': f"output_{dataset}_embeddings3.parquet"
        },
        'COLUMNS': {
            'timestamp': 'timestamp',      
            'size': 'packet_length',                
            'src_ip': 'src_ip',
            'dst_ip': 'dst_ip',
            'src_port': 'src_port',
            'dst_port': 'dst_port',
            'protocol': 'protocol',
            'label_multi': 'label',      
            'label_binary': 'label_type' 
        },
        'HYPERPARAMETERS': {
            'BATCH_SIZE': 50000 
        }
    }
    
    if os.path.exists(PAINEL_CONFIG['IO']['INPUT_FILE']):
       run_pipeline(PAINEL_CONFIG)
    else:
        print(f"Arquivo de entrada não encontrado: {PAINEL_CONFIG['IO']['INPUT_FILE']}")