import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, 
    KFold, 
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report
)
from DataLoader import DataLoader
from PreProcessor import Preprocessor
from NearMiss import NearMissResampler

def main():
    # 1. Tratamento de dados
    print("=" * 60)
    print("1. CARREGAMENTO E PREPROCESSAMENTO DOS DADOS")
    print("=" * 60)

    # Carregar dados
    loader = DataLoader(delimiter=';')
    data, header = loader.load('dataset/bank.csv')
    print(f"Dados carregados: {data.shape[0]} amostras, {data.shape[1]} features")
    print(f"Features: {header}\n")

    # Preprocessar dados
    preprocessor = Preprocessor()
    X, y = preprocessor.transform(data)
    print(f"Dados processados: X shape {X.shape}, y shape {y.shape}")

    # Equilibrar dados usando Near Miss
    print("\nEquilibrando dataset com NearMiss...")
    resampler = NearMissResampler(version=3, n_neighbors=3, m_neighbors=3)
    X_balanced, y_balanced = resampler.resample(X, y)
    print(f"Dados balanceados: {X_balanced.shape[0]} amostras")

    unique, counts = np.unique(y_balanced, return_counts=True)
    print(f"Distribuição de classes: {dict(zip(unique, counts))}\n")

    # 2. Divisão treino/teste
    print("=" * 60)
    print("2. DIVISÃO TREINO/TESTE")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras\n")

    # 3. Validação cruzada com K-Fold
    print("=" * 60)
    print("3. VALIDAÇÃO CRUZADA (K-FOLD)")
    print("=" * 60)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_macro')

    print(f"Modelo: Random Forest com 100 árvores")
    print(f"Estratégia: K-Fold com 5 splits")
    print(f"F1-Score (Macro) por fold: {cv_scores}")
    print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    # 4. Treinamento do modelo
    print("=" * 60)
    print("4. TREINAMENTO DO MODELO")
    print("=" * 60)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!\n")

    # 5. Avaliação no conjunto de teste
    print("=" * 60)
    print("5. AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Acurácia:  {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Feature Importance
    print("=" * 60)
    print("6. IMPORTÂNCIA DAS FEATURES")
    print("=" * 60)
    feature_importance = model.feature_importances_
    feature_names = [f"Feature_{i}" for i in range(len(header) - 1)]

    sorted_idx = np.argsort(feature_importance)[-10:]
    print(f"Top 10 features mais importantes:")
    for idx in sorted_idx[::-1]:
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

if __name__ == "__main__":
    main()