import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def run_multiclass_classification(
    train_path: str,
    test_path: str | None = None,
    label_col: str = "Label",
    n_estimators: int = 100,
    random_state: int = 42,
    min_samples_per_class: int = 3
):
    print("="*80)
    print("CLASSIFICAÇÃO MULTICLASSE (RF) - FEATURES + MÉTRICAS + AUC")
    print("="*80)

    # --- 1. CARREGAMENTO E FILTRAGEM ---
    if test_path:
        print(f"Modo: Arquivos Separados.")
        df_train = pd.read_parquet(train_path)
        df_test  = pd.read_parquet(test_path)
    else:
        print(f"Modo: Arquivo Único (Split 70/30).")
        try:
            df_full = pd.read_parquet(train_path)
        except Exception as e:
            print(f"[x] Erro crítico ao ler arquivo: {e}")
            return

        if label_col not in df_full.columns:
            print(f"[x] Coluna '{label_col}' não encontrada.")
            return

        # Filtra classes raras (< 3 amostras)
        counts = df_full[label_col].value_counts()
        rare = counts[counts < min_samples_per_class].index
        if len(rare) > 0:
            print(f"⚠️  Removendo classes raras (<{min_samples_per_class} amostras): {list(rare)}")
            df_full = df_full[~df_full[label_col].isin(rare)]

        # Split
        try:
            df_train, df_test = train_test_split(
                df_full, test_size=0.30, random_state=random_state, stratify=df_full[label_col]
            )
        except ValueError:
            print("⚠️  Falha no stratify (classes muito desbalanceadas). Usando split aleatório.")
            df_train, df_test = train_test_split(df_full, test_size=0.30, random_state=random_state)

    # --- 2. PREPARAÇÃO (X, y) ---
    drop_cols = [
        label_col, 'Label_Type', 'label_type', 'src_packet_indices', 'Flow_ID', 
        'global_index', 'timestamp', 'src_ip', 'dst_ip', 'protocol', 
        'src_port', 'dst_port', 'Packet_Count', 'encoded_label', 
        'payload_size', 'ttl', 'tos', 'node_A_port', 'node_B_port', 'Service_Port'
    ]
    
    # drop_cols = [
    #     label_col, 'Label_Type', 'label_type', 'src_packet_indices', 'Flow_ID', 
    #     'global_index', 'src_ip', 'dst_ip', 'protocol', 
    #     'src_port', 'dst_port', 'Packet_Count', 'encoded_label', 
    #     'payload_size', 'ttl', 'tos'
    # ] #size + ts
    
    
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
    X_test  = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
    
    # Garante numéricos
    X_train = X_train.select_dtypes(include=['number'])
    X_test  = X_test.select_dtypes(include=['number'])

    # <--- NOVO: Print das Features --->
    # Capturamos os nomes antes do Imputer transformar em array numpy
    feature_names = X_train.columns.tolist()
    print(f"\n🔍 Features selecionadas para o modelo ({len(feature_names)}):")
    print(feature_names)
    print("-" * 40)

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[label_col].astype(str))
    
    # Ajuste para teste (Labels novos viram "Unknown" ou são filtrados)
    y_test_str = df_test[label_col].astype(str)
    # Filtra labels do teste que não existem no treino
    mask_known = y_test_str.isin(le.classes_)
    if not mask_known.all():
        print(f"⚠️  Ignorando { (~mask_known).sum() } amostras de teste com classes desconhecidas.")
        X_test = X_test[mask_known]
        y_test_str = y_test_str[mask_known]
    
    y_test = le.transform(y_test_str)

    # Imputer
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    # --- 3. TREINAMENTO ---
    print(f"\n🚀 Treinando Modelo (Classes: {len(le.classes_)}) ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # --- 4. CÁLCULO DE MÉTRICAS ---
    print("📊 Calculando métricas...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Métricas Globais
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC AUC
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"⚠️  Não foi possível calcular AUC: {e}")
        auc = 0.0

    print("\n" + "="*40)
    print("MÉTRICAS GLOBAIS (Ponderadas)")
    print("="*40)
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"📡 Recall:    {rec:.4f}")
    print(f"⚖️  F1 Score:  {f1:.4f}")
    print(f"⭐ ROC AUC:   {auc:.4f}")
    print("="*40)

    # Relatório Detalhado
    target_names = [str(c) for c in le.classes_]
    print("\n--- Relatório Detalhado por Classe ---")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Matriz de Confusão
    if len(target_names) <= 15:
        print("\n--- Matriz de Confusão ---")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm, index=target_names, columns=target_names))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "model": clf,
        "features": feature_names # Também retorno a lista caso queira usar depois
    }

if __name__ == "__main__":
    dataset = "5gnidd"
    INPUT_FILE = f"output_{dataset}_service3/output_{dataset}_embeddings_service3.parquet"
    # INPUT_FILE = f"datasets/{dataset}/all_{dataset}.parquet"
    run_multiclass_classification(
        train_path=INPUT_FILE,
        label_col="Label",
        min_samples_per_class=3
    )