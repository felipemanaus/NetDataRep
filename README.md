# NetDataRep: A Multilevel Method for Network Traffic Representation

**Artigo Relacionado:**
* **Título:** Rethinking Network Traffic Representation: A Multilevel Method for Traffic Analysis
* **Autores:** Felipe Melo, Fernando Nakayama, Michele Nogueira (UFMG)
* **Referência:** SIGCOMM 2026

---

## Descrição do Projeto

Este repositório contém a implementação do **NetDataRep**, um método multinível (multilevel) focado na representação explícita de tráfego de rede. O objetivo do método é gerar vetores numéricos de alta densidade semântica (embeddings) a partir de fluxos brutos, preservando a estrutura temporal e as informações relacionais do tráfego.

Diferente das abordagens baseadas em Machine Learning que dependem de quantidades enormes de dados (*Raw Packets*) e modelos pesados, o NetDataRep processa o tráfego em três diferentes níveis de granulação:
1. **Packets per Service:** Agrupa por IP e Porta de Destino (Ideal para capturar padrões orientados à aplicação / serviço).
2. **Packets per Flow (5-Tuple):** Agrupa unidirecionalmente (A → B) preservando a granularidade precisa da sessão.
3. **Packets per Biflow:** Sintetiza o diálogo unindo tráfego bidirecional (A ↔ B) para capturar o comportamento de *request-response*.

Estas representações alcançam precisões altíssimas de detecção e classificação (chegando a 99.99%) enquanto descartam "atalhos" (shortcuts) que prejudicam a generalização dos modelos, permitindo rodar classificadores de aprendizado de máquina simples com um fragmento minúsculo dos dados originais (em alguns casos, apenas 1% do volume).

---

## Estrutura do Repositório

### 1. Geração de Representações (Feature Extractors)
Implementamos os scripts responsáveis pela conversão dos pacotes para as matrizes matemáticas de Extração e Agrupamento (Grouping).
* `embeddings_flow.py` : Implementa a extração no nível **Unidirecional (Flow / 5-tuple)**.
* `embeddings_service.py`: Implementa a extração no nível de **Serviço (Service / Destination Port)**.
* `embeddings_biflow.py`: Implementa a extração no nível **Bidirecional (Biflow / A ↔ B)** através da normalização da chave de rede.

**As propriedades matemáticas capturadas incluem:**
* **Características de Distribuição:** Estatísticas básicas de Packet Length, Timestamp Relativo e IAT (Inter-Arrival Time).
* **Características Físicas e Informacionais:** Norma L2, Entropia de Shannon e Índice de Dispersão (Burstiness).
* **Características Espectrais:** FFT (Fast Fourier Transform) computando frequência dominante.
* **Dinâmicas Estruturais e Temporais:** PCA e Coeficientes de transição linear extraídos via *Vector Autoregression* (VAR).

### 2. Classificação / Avaliação
* `classification.py`: Contém a lógica base de validação usando um **Random Forest Classifier** conforme a experimentação descrita no artigo. Mede Acurácia, F1-Score, AUC, Precisão e exibe detalhes de importância preditiva em Cenários Multiclasse.

---

## Pré-requisitos e Instalação

São recomendadas versões do Python >= 3.8.

```bash
pip install pandas numpy scipy tqdm scikit-learn pyarrow fastparquet
```

---

## Como Executar (Guia de Reprodução)

### Passo 1: Geração das Representações (NetDataRep Embeddings)
O método baseia-se num pipeline que pega um dataset de pacotes formatado em `.parquet` (necessita das colunas básicas de Timestamp, IP/Ports, e Tamanho de Pacote) e os compacta na estratégia desejada.

1. Configure as variáveis `dataset` e de entrada/saída no bloco `if __name__ == "__main__":` do script desejado.
2. Execute a estratégia desejada para gerar o arquivo `.parquet` do embedding correspondente.

**Opção A - Packets per Flow (Unidirecional):**
```bash
python embeddings_flow.py
```

**Opção B - Packets per Service (Apenas Destino):**
```bash
python embeddings_service.py
```

**Opção C - Packets per Biflow (Bidirecional):**
```bash
python embeddings_biflow.py
```

O script salvará o resultado (ex: `output_edgeiiot_embeddings_service3.parquet`) em um diretório configurado no output.

### Passo 2: Classificação e Inferência
De posse do embedding gerado, o teste base avaliará as representações:

1. Edite o script `classification.py` modificando o `INPUT_FILE` para apontar para o `.parquet` que você gerou no Passo 1.
2. Execute o modelo:
```bash
python classification.py
```
3. A saída mostrará a performance (Accuracy, Precision, Recall, F1 e AUC-ROC) e listará as 40 dimensões numéricas padronizadas (variáveis de saída) usadas para a tomada de decisão, demonstrando que o algoritmo não "decorou" os IPs originais para predizer o resultado (shortcut learning problem).

---

## Resultados em Destaque no Artigo
O método avaliado em datasets heterogêneos como *AWID3, 5G-NIDD, UNSW-NB15 e Edge-IIoTset* mostra:
* Classificações (Accuracy e F1-Scores) superiores a **99%** usando o Random Forest, removendo os falsos positivos causados pelas regras temporais ou de IPs do Dataset Original.
* Processamento e latência de processamento incrivelmente eficientes com tempo médio variando de $0.0024ms/pkt$ (Service) a $pprox0.0149ms/pkt$ (Biflow), atendendo aos requisitos estritos de IoT/5G para monitoramento em tempo real em $O(N)$ de complexidade de escalonamento.

---
