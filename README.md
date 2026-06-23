# Bank Marketing Analysis

Projeto de análise e predição de campanhas de marketing bancário usando Machine Learning.

## 📋 Objetivo

Classificar clientes que aceitarão ou recusarão uma oferta de depósito a prazo (target variável `y`).

## 📁 Estrutura do Projeto

- **DataLoader.py** - Carrega dados CSV do dataset
- **PreProcessor.py** - Normaliza features e codifica variáveis categóricas
- **NearMiss.py** - Balanceia classes desbalanceadas usando undersampling
- **main.py** - Pipeline completo de treino e avaliação
- **dataset/** - Contém os arquivos CSV (bank.csv e bank-full.csv)

## 🚀 Como Usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o projeto
```bash
python main.py
```

## 📊 Pipeline de Processamento

1. **Carregamento**: Lê 45.211 amostras com 17 features (dataset completo)
2. **Preprocessamento**: Normaliza dados numéricos e codifica categóricos
3. **Balanceamento**: NearMiss V3 reduz para 10.578 amostras balanceadas
4. **Divisão**: 80% treino (8.462), 20% teste (2.116)
5. **Validação**: K-Fold com 5 splits (F1: 74.21%)
6. **Treinamento**: Random Forest com 100 árvores
7. **Avaliação**: Métricas completas e feature importance

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Acurácia | 74.86% |
| Precisão | 74.97% |
| Recall | 74.86% |
| F1-Score | 74.83% |

**Top 3 Features mais importantes:**
- Feature_11 (duration) - 21.57%
- Feature_5 (balance) - 12.70%
- Feature_0 (age) - 10.17%

## 📦 Dependências

- numpy
- scikit-learn

## 👤 Autor

Trabalho acadêmico de Machine Learning
