#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para converter seu randomforest.py existente para a nova estrutura.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel
from utils.logger import setup_logger


def analyze_existing_code():
    """Analisa seu código existente e sugere adaptações."""
    
    # Ler seu arquivo atual
    try:
        with open('randomforest.py', 'r', encoding='utf-8') as f:
            existing_code = f.read()
            
        print("📄 CÓDIGO ATUAL ENCONTRADO:")
        print("-" * 40)
        print(existing_code)
        print("-" * 40)
        
        # Analisar o que já existe
        has_pandas = 'import pandas' in existing_code
        has_sklearn = 'sklearn' in existing_code
        has_data_loading = 'read_csv' in existing_code or 'load' in existing_code
        
        print("\n🔍 ANÁLISE DO CÓDIGO ATUAL:")
        print(f"✅ Pandas importado: {has_pandas}")
        print(f"✅ Scikit-learn usado: {has_sklearn}")
        print(f"✅ Carregamento de dados: {has_data_loading}")
        
        return existing_code
        
    except FileNotFoundError:
        print("❌ Arquivo randomforest.py não encontrado no diretório atual")
        return None


def create_converted_example():
    """Cria exemplo de conversão baseado no seu código."""
    
    converted_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEU CÓDIGO CONVERTIDO - Random Forest com nova estrutura.

Este arquivo mostra como seu código randomforest.py seria
na nova estrutura profissional.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

# Importar classes da nova estrutura
from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel
from utils.logger import setup_logger


def main():
    """Função principal - seu código convertido."""
    
    # ===== CONFIGURAÇÃO =====
    logger = setup_logger(name='converted_rf', level='INFO')
    logger.info("Iniciando Random Forest - Versão Convertida")
    
    # ===== ONDE ESTAVA: import pandas as pd =====
    # AGORA: Usar DataLoader para carregamento robusto
    
    try:
        # 1. CARREGAR DADOS (substituir por seu dataset)
        logger.info("Carregando dados")
        data_loader = DataLoader(data_path="data/raw")
        
        # TODO: Substituir pelo seu arquivo real
        # df = data_loader.load_csv("seu_dataset.csv")
        
        # Para demonstração, criar dados de exemplo
        # (REMOVA esta parte e use seus dados reais)
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randint(0, 10, n_samples),
            'categoria_A': np.random.choice(['Tipo1', 'Tipo2', 'Tipo3'], n_samples),
            'categoria_B': np.random.choice(['X', 'Y'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Adicionar alguns valores missing para testar
        missing_idx = np.random.choice(df.index, size=50, replace=False)
        df.loc[missing_idx, 'feature_1'] = np.nan
        # ===== FIM DOS DADOS DE EXEMPLO =====
        
        logger.info(f"Dataset carregado: {df.shape}")
        print(f"📊 Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
        print(f"📋 Colunas: {list(df.columns)}")
        
        # Verificar target
        if 'target' not in df.columns:
            raise ValueError("Coluna 'target' não encontrada. Modifique TARGET_COLUMN no script.")
            
        # 2. PRÉ-PROCESSAMENTO AUTOMÁTICO
        logger.info("Pré-processamento automático")
        preprocessor = DataPreprocessor()
        
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column='target',
            missing_strategy='mean',
            encoding_type='onehot',
            scale_features=True,
            test_size=0.2
        )
        
        print(f"✅ Dados processados:")
        print(f"   🏋️ Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"   🧪 Teste: {X_test.shape[0]} amostras")
        
        # 3. TREINAMENTO
        logger.info("Treinamento do modelo")
        model = RandomForestModel(
            task_type='classification',
            n_estimators=100,
            random_state=42
        )
        
        print("🚀 Treinando Random Forest...")
        
        # Busca de hiperparâmetros (opcional)
        if True:  # Substituir por TUNE_HYPERPARAMS
            print("🔍 Buscando melhores hiperparâmetros...")
            tuning_results = model.hyperparameter_tuning(X_train, y_train)
            print(f"✅ Melhores parâmetros encontrados!")
            
            # Mostrar alguns parâmetros principais
            best_params = tuning_results['best_params']
            print(f"   🌳 n_estimators: {best_params.get('n_estimators', 'default')}")
            print(f"   📏 max_depth: {best_params.get('max_depth', 'default')}")
        else:
            model.train(X_train, y_train)
            
        # 4. AVALIAÇÃO
        logger.info("Avaliando modelo")
        metrics = model.evaluate(X_test, y_test)
        
        print("\\n📊 MÉTRICAS DE PERFORMANCE:")
        print("-" * 30)
        print(f"🎯 Acurácia: {metrics['accuracy']:.4f}")
        print(f"📏 Precisão: {metrics['precision']:.4f}")
        print(f"🔍 Recall: {metrics['recall']:.4f}")
        print(f"⚖️ F1-Score: {metrics['f1']:.4f}")
        
        # Validação cruzada
        cv_results = model.cross_validate(X_train, y_train)
        print(f"✅ Validação Cruzada: {cv_results['mean_score']:.4f} ± {cv_results['std_score']*2:.4f}")
        
        # 5. FEATURE IMPORTANCE
        top_features = model.get_feature_importance(top_n=10)
        print("\n🎯 FEATURES MAIS IMPORTANTES:")
        print("-" * 35)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        # 6. SALVAR MODELO E RESULTADOS
        logger.info("Salvando modelo e resultados")
        Path("data/models").mkdir(parents=True, exist_ok=True)
        
        model_path = f"data/models/random_forest_classification.joblib"
        preprocessor_path = f"data/models/preprocessor_classification.joblib"
        
        model.save_model(model_path)
        preprocessor.save_preprocessors(preprocessor_path)
        
        # Salvar métricas em JSON
        import json
        metrics_data = {
            **{k: v for k, v in metrics.items() if k not in ['classification_report']},
            'cv_results': cv_results,
            'feature_importance': top_features.to_dict('records')
        }
        
        if 'tuning_results' in locals():
            metrics_data['best_params'] = tuning_results['best_params']
        
        # Converter numpy arrays para listas
        for key, value in metrics_data.items():
            if isinstance(value, np.ndarray):
                metrics_data[key] = value.tolist()
        
        metrics_path = f"data/models/metrics_classification.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   🤖 Modelo: {model_path}")
        print(f"   🔧 Preprocessador: {preprocessor_path}")
        print(f"   📊 Métricas: {metrics_path}")
        
        # 7. DEMONSTRAR PREDIÇÕES
        print("\n🔮 TESTANDO PREDIÇÕES:")
        
        # Exemplo de predição com novos dados
        sample_data = pd.DataFrame({
            'feature_1': [0.5],
            'feature_2': [-0.2],
            'feature_3': [5],
            'categoria_A': ['Tipo1'],
            'categoria_B': ['X']
        })
        
        # Processar da mesma forma que no treinamento
        sample_processed = preprocessor.handle_missing_values(sample_data)
        sample_encoded = preprocessor.encode_categorical_features(sample_processed)
        sample_scaled = preprocessor.scale_features(sample_encoded)
        
        # Fazer predição
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)
        
        print(f"   📍 Predição: {prediction[0]}")
        print(f"   📊 Probabilidades: Classe 0: {probabilities[0][0]:.3f}, Classe 1: {probabilities[0][1]:.3f}")
        
        print("\n" + "="*50)
        print("🎊 CONVERSÃO CONCLUÍDA!")
        print("="*50)
        
        print("\n🚀 EXECUTE O DASHBOARD:")
        print("python scripts/run_dashboard.py")
        print("\nOu com Docker:")
        print("docker-compose up --build")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        print(f"\n❌ ERRO: {str(e)}")
        return False


def show_migration_guide():
    """Mostra guia de migração passo a passo."""
    
    print("\n📋 GUIA DE MIGRAÇÃO DO SEU CÓDIGO:")
    print("="*50)
    
    print("\n🔄 PASSO A PASSO:")
    print("1. 📁 Coloque seu dataset em data/raw/")
    print("2. ✏️  Modifique as variáveis no início deste script:")
    print("   - DATASET_FILENAME")
    print("   - TARGET_COLUMN") 
    print("   - TASK_TYPE")
    print("3. 🚀 Execute: python convert_existing_code.py")
    print("4. 📱 Execute: python scripts/run_dashboard.py")
    
    print("\n🔧 ONDE SEU CÓDIGO ATUAL SE ENCAIXA:")
    print("┌─────────────────────────┬─────────────────────────┐")
    print("│ SEU CÓDIGO ATUAL        │ NOVA ESTRUTURA          │")
    print("├─────────────────────────┼─────────────────────────┤")
    print("│ import pandas as pd     │ DataLoader class        │")
    print("│ pd.read_csv()           │ data_loader.load_csv()  │")
    print("│ Manual preprocessing    │ DataPreprocessor class  │")
    print("│ RandomForestClassifier  │ RandomForestModel class │")
    print("│ Manual evaluation       │ model.evaluate()        │")
    print("│ joblib.dump()           │ model.save_model()      │")
    print("└─────────────────────────┴─────────────────────────┘")
    
    print("\n💡 BENEFÍCIOS DA CONVERSÃO:")
    print("✅ Código mais limpo e organizad")
    print("✅ Tratamento automático de dados categóricos")
    print("✅ Busca automática de hiperparâmetros")
    print("✅ Dashboard visual interativo")
    print("✅ Logging e monitoramento")
    print("✅ Pronto para Docker e produção")
    print("✅ Testes automatizados")


if __name__ == "__main__":
    # Verificar se arquivo existe
    if Path('randomforest.py').exists():
        existing_code = analyze_existing_code()
        
        print("\n" + "="*50)
        print("🎯 QUER EXECUTAR A CONVERSÃO?")
        print("="*50)
        
        response = input("\nDigite 'sim' para converter automaticamente: ").lower().strip()
        
        if response in ['sim', 's', 'yes', 'y']:
            print("\n🔄 Executando conversão...")
            success = main()
            
            if success:
                print("\n✅ Seu código foi convertido com sucesso!")
                print("📱 Execute agora: python scripts/run_dashboard.py")
            else:
                print("\n❌ Erro na conversão. Verifique os dados e tente novamente.")
        else:
            show_migration_guide()
            print("\n📝 QUANDO ESTIVER PRONTO:")
            print("1. Ajuste as configurações no início deste script")
            print("2. Execute: python convert_existing_code.py")
    else:
        print("📄 Arquivo randomforest.py não encontrado")
        show_migration_guide()
        
        print("\n🆕 COMEÇAR DO ZERO?")
        print("Execute: python quick_start.py")


# Função main do exemplo anterior
def main():
    """Função de conversão (mesmo código do quick_start adaptado)."""
    
    logger = setup_logger(name='conversion', level='INFO')
    
    try:
        # Dados de exemplo para demonstração
        logger.info("Criando dados de exemplo (substitua pelos seus)")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'idade': np.random.randint(18, 80, n_samples),
            'renda': np.random.normal(50000, 15000, n_samples),
            'experiencia': np.random.randint(0, 40, n_samples),
            'educacao': np.random.choice(['Fundamental', 'Médio', 'Superior'], n_samples),
            'genero': np.random.choice(['M', 'F'], n_samples),
            'aprovado': np.random.randint(0, 2, n_samples)
        })
        
        print(f"📊 Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Pré-processamento
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column='aprovado',
            test_size=0.2
        )
        
        print(f"✅ Pré-processamento: {X_train.shape[1]} features finais")
        
        # Treinamento
        model = RandomForestModel(task_type='classification')
        tuning_results = model.hyperparameter_tuning(X_train, y_train)
        
        # Avaliação
        metrics = model.evaluate(X_test, y_test)
        cv_results = model.cross_validate(X_train, y_train)
        
        print(f"\n📊 RESULTADOS:")
        print(f"🎯 Acurácia: {metrics['accuracy']:.4f}")
        print(f"✅ CV Score: {cv_results['mean_score']:.4f}")
        
        # Salvar
        Path("data/models").mkdir(parents=True, exist_ok=True)
        model.save_model("data/models/random_forest_classification.joblib")
        preprocessor.save_preprocessors("data/models/preprocessor_classification.joblib")
        
        # Salvar métricas
        import json
        metrics_data = {
            **{k: v for k, v in metrics.items() if k not in ['classification_report']},
            'cv_results': cv_results,
            'feature_importance': model.get_feature_importance().to_dict('records'),
            'best_params': tuning_results['best_params']
        }
        
        with open("data/models/metrics_classification.json", 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        print("💾 Modelo salvo em data/models/")
        return True
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        return False