#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para migrar código existente para a nova estrutura.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel
from utils.logger import setup_logger
import pandas as pd


def convert_old_code_example():
    """
    Exemplo de como converter código do Jupyter Notebook para a nova estrutura.
    
    Substitua este exemplo pelo seu código atual do randomforest.py
    """
    
    # Configurar logging
    logger = setup_logger(name='migration', level='INFO')
    logger.info("Iniciando migração do código")
    
    try:
        # ===== PASSO 1: CARREGAR DADOS =====
        # ANTES (seu código atual):
        # import pandas as pd
        # df = pd.read_csv('seu_dataset.csv')
        
        # DEPOIS (nova estrutura):
        logger.info("Carregando dados com DataLoader")
        data_loader = DataLoader(data_path="data/raw")
        
        # Exemplo: substitua por seu arquivo real
        # df = data_loader.load_csv("seu_dataset.csv")
        
        # Para teste, vou criar dados fictícios (REMOVA ESTA PARTE)
        # ===== DADOS DE EXEMPLO (REMOVER) =====
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'idade': np.random.randint(18, 80, n_samples),
            'salario': np.random.normal(50000, 15000, n_samples),
            'experiencia': np.random.randint(0, 40, n_samples),
            'educacao': np.random.choice(['Fundamental', 'Médio', 'Superior'], n_samples),
            'sexo': np.random.choice(['M', 'F'], n_samples),
            'aprovado': np.random.randint(0, 2, n_samples)  # target
        })
        # ===== FIM DOS DADOS DE EXEMPLO =====
        
        logger.info(f"Dataset carregado: {df.shape}")
        
        # ===== PASSO 2: PRÉ-PROCESSAMENTO =====
        # ANTES (código manual):
        # from sklearn.preprocessing import StandardScaler, LabelEncoder
        # from sklearn.model_selection import train_test_split
        # ... código manual de preprocessing ...
        
        # DEPOIS (nova estrutura):
        logger.info("Iniciando pré-processamento")
        preprocessor = DataPreprocessor()
        
        # Pipeline automático
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column='aprovado',  # Substitua pela sua coluna target
            missing_strategy='mean',
            encoding_type='onehot',
            scale_features=True,
            test_size=0.2
        )
        
        logger.info(f"Dados processados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
        # ===== PASSO 3: TREINAMENTO =====
        # ANTES (código básico):
        # from sklearn.ensemble import RandomForestClassifier
        # rf = RandomForestClassifier()
        # rf.fit(X_train, y_train)
        
        # DEPOIS (nova estrutura):
        logger.info("Criando e treinando modelo")
        model = RandomForestModel(
            task_type='classification',  # ou 'regression'
            n_estimators=100,
            random_state=42
        )
        
        # Treinamento com busca de hiperparâmetros (opcional)
        tuning_results = model.hyperparameter_tuning(X_train, y_train)
        logger.info(f"Melhores parâmetros: {tuning_results['best_params']}")
        
        # ===== PASSO 4: AVALIAÇÃO =====
        # ANTES (avaliação básica):
        # from sklearn.metrics import accuracy_score
        # predictions = rf.predict(X_test)
        # accuracy = accuracy_score(y_test, predictions)
        
        # DEPOIS (avaliação completa):
        logger.info("Avaliando modelo")
        metrics = model.evaluate(X_test, y_test)
        
        # Validação cruzada
        cv_results = model.cross_validate(X_train, y_train)
        
        # Log dos resultados
        logger.info("RESULTADOS:")
        logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']*2:.4f}")
        
        # ===== PASSO 5: FEATURE IMPORTANCE =====
        # ANTES (básico):
        # importance = rf.feature_importances_
        
        # DEPOIS (estruturado):
        top_features = model.get_feature_importance(top_n=10)
        logger.info("TOP 10 FEATURES:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # ===== PASSO 6: SALVAR MODELO =====
        # ANTES (básico):
        # import joblib
        # joblib.dump(rf, 'model.pkl')
        
        # DEPOIS (estruturado):
        Path("data/models").mkdir(parents=True, exist_ok=True)
        model.save_model("data/models/random_forest_classification.joblib")
        preprocessor.save_preprocessors("data/models/preprocessor_classification.joblib")
        
        # Salvar métricas
        import json
        metrics_data = {
            **metrics,
            'cv_results': cv_results,
            'feature_importance': top_features.to_dict('records'),
            'best_params': tuning_results['best_params']
        }
        
        # Converter arrays numpy para listas (JSON serializable)
        for key, value in metrics_data.items():
            if isinstance(value, np.ndarray):
                metrics_data[key] = value.tolist()
        
        with open("data/models/metrics_classification.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Modelo e métricas salvos com sucesso!")
        
        # ===== PASSO 7: DEMONSTRAR PREDIÇÕES =====
        # ANTES (básico):
        # new_prediction = rf.predict([[valores...]])
        
        # DEPOIS (estruturado):
        logger.info("Testando predições")
        
        # Exemplo de predição
        sample_data = pd.DataFrame({
            'idade': [35],
            'salario': [60000],
            'experiencia': [10],
            'educacao': ['Superior'],
            'sexo': ['M']
        })
        
        # Processar dados da mesma forma que no treinamento
        sample_processed = preprocessor.handle_missing_values(sample_data)
        sample_encoded = preprocessor.encode_categorical_features(sample_processed)
        sample_scaled = preprocessor.scale_features(sample_encoded)
        
        # Fazer predição
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)
        
        logger.info(f"Predição: {prediction[0]}")
        logger.info(f"Probabilidades: {probabilities[0]}")
        
        logger.info("✅ Migração concluída com sucesso!")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro durante migração: {str(e)}")
        return False


def show_comparison():
    """Mostra comparação entre código antigo e novo."""
    
    print("\n" + "="*60)
    print("📊 COMPARAÇÃO: ANTES vs DEPOIS")
    print("="*60)
    
    print("\n🔴 ANTES (Jupyter/Script simples):")
    print("""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar dados
df = pd.read_csv('dataset.csv')

# Preprocessing manual
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treinar modelo
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Avaliar
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
""")
    
    print("\n🟢 DEPOIS (Estrutura profissional):")
    print("""
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.random_forest import RandomForestModel

# Carregar dados
loader = DataLoader()
df = loader.load_csv('dataset.csv')

# Pipeline de preprocessing
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df, 'target')

# Treinar com busca de hiperparâmetros
model = RandomForestModel(task_type='classification')
model.hyperparameter_tuning(X_train, y_train)

# Avaliação completa
metrics = model.evaluate(X_test, y_test)
cv_results = model.cross_validate(X_train, y_train)

# Salvar tudo
model.save_model('data/models/model.joblib')
preprocessor.save_preprocessors('data/models/preprocessor.joblib')
""")
    
    print("\n💎 BENEFÍCIOS DA NOVA ESTRUTURA:")
    print("✅ Código reutilizável e modular")
    print("✅ Tratamento automático de dados categóricos")
    print("✅ Busca automática de hiperparâmetros")
    print("✅ Métricas completas e logging")
    print("✅ Dashboard visual interativo")
    print("✅ Pronto para Docker e produção")
    print("✅ Pipeline de CI/CD automático")
    print("✅ Testes automatizados")


if __name__ == "__main__":
    import numpy as np
    
    print("🔄 Executando migração de exemplo...")
    success = convert_old_code_example()
    
    if success:
        show_comparison()
        print("\n🎯 COMO MIGRAR SEU CÓDIGO:")
        print("1. Substitua os dados de exemplo pelos seus dados reais")
        print("2. Ajuste o nome da coluna target")
        print("3. Execute: python migrate_code.py")
        print("4. Execute: python scripts/run_dashboard.py")
    else:
        print("❌ Erro na migração. Verifique os logs acima.")
