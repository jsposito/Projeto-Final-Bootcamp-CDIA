#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de início rápido - converte seu randomforest.py atual para a nova estrutura.

INSTRUÇÕES:
1. Coloque seu dataset CSV em data/raw/
2. Modifique as variáveis abaixo com seus dados reais
3. Execute: python quick_start.py
"""

import sys
from pathlib import Path
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel
from utils.logger import setup_logger


# ==========================================
# 🔧 CONFIGURAÇÕES - MODIFIQUE AQUI
# ==========================================

# Nome do seu arquivo de dados (deve estar em data/raw/)
DATASET_FILENAME = "seu_dataset.csv"  # ← MODIFIQUE AQUI

# Nome da coluna target (a que você quer prever)
TARGET_COLUMN = "target"  # ← MODIFIQUE AQUI

# Tipo de problema ('classification' ou 'regression')
TASK_TYPE = "classification"  # ← MODIFIQUE AQUI

# Proporção para teste (20% é padrão)
TEST_SIZE = 0.2

# Se deve fazer busca de hiperparâmetros (True/False)
TUNE_HYPERPARAMS = True

# ==========================================


def main():
    """Conversão automática do seu código."""
    
    # Setup inicial
    logger = setup_logger(name='quick_start', level='INFO')
    
    print("🌲 QUICK START - Random Forest ML Project")
    print("=" * 50)
    
    try:
        # ===== 1. VERIFICAR SE DATASET EXISTE =====
        data_path = Path("data/raw") / DATASET_FILENAME
        
        if not data_path.exists():
            print(f"❌ Erro: Dataset não encontrado em {data_path}")
            print("\n📋 INSTRUÇÕES:")
            print("1. Coloque seu arquivo CSV em data/raw/")
            print("2. Modifique DATASET_FILENAME no início deste script")
            print("3. Execute novamente: python quick_start.py")
            return False
            
        print(f"✅ Dataset encontrado: {data_path}")
        
        # ===== 2. CARREGAR E VALIDAR DADOS =====
        logger.info("Carregando dados")
        data_loader = DataLoader()
        df = data_loader.load_csv(DATASET_FILENAME)
        
        # Mostrar info básica
        print(f"📊 Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
        print(f"📋 Colunas: {list(df.columns)}")
        
        # Verificar se target existe
        if TARGET_COLUMN not in df.columns:
            print(f"❌ Erro: Coluna target '{TARGET_COLUMN}' não encontrada")
            print(f"📋 Colunas disponíveis: {list(df.columns)}")
            print("💡 Modifique TARGET_COLUMN no início do script")
            return False
            
        print(f"✅ Coluna target encontrada: {TARGET_COLUMN}")
        
        # ===== 3. PRÉ-PROCESSAMENTO AUTOMÁTICO =====
        logger.info("Iniciando pré-processamento automático")
        preprocessor = DataPreprocessor()
        
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column=TARGET_COLUMN,
            missing_strategy='mean',
            encoding_type='onehot',
            scale_features=True,
            test_size=TEST_SIZE
        )
        
        print(f"✅ Pré-processamento concluído:")
        print(f"   📈 Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"   📊 Teste: {X_test.shape[0]} amostras")
        
        # ===== 4. TREINAMENTO DO MODELO =====
        logger.info("Criando modelo Random Forest")
        model = RandomForestModel(
            task_type=TASK_TYPE,
            n_estimators=100,
            random_state=42
        )
        
        if TUNE_HYPERPARAMS:
            print("🔍 Buscando melhores hiperparâmetros...")
            tuning_results = model.hyperparameter_tuning(X_train, y_train)
            print(f"✅ Melhores parâmetros: {tuning_results['best_params']}")
        else:
            print("🚀 Treinando modelo com parâmetros padrão...")
            model.train(X_train, y_train)
        
        # ===== 5. AVALIAÇÃO COMPLETA =====
        logger.info("Avaliando modelo")
        metrics = model.evaluate(X_test, y_test)
        cv_results = model.cross_validate(X_train, y_train)
        
        # Exibir resultados
        print("\n📊 RESULTADOS DA AVALIAÇÃO:")
        print("-" * 30)
        
        if TASK_TYPE == 'classification':
            print(f"🎯 Acurácia: {metrics['accuracy']:.4f}")
            print(f"📏 Precisão: {metrics['precision']:.4f}")
            print(f"🔍 Recall: {metrics['recall']:.4f}")
            print(f"⚖️ F1-Score: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"📈 ROC-AUC: {metrics['roc_auc']:.4f}")
        else:
            print(f"📊 R² Score: {metrics['r2']:.4f}")
            print(f"📏 RMSE: {metrics['rmse']:.4f}")
            print(f"📐 MAE: {metrics['mae']:.4f}")
            
        print(f"✅ Validação Cruzada: {cv_results['mean_score']:.4f} ± {cv_results['std_score']*2:.4f}")
        
        # ===== 6. FEATURE IMPORTANCE =====
        top_features = model.get_feature_importance(top_n=10)
        print("\n🎯 TOP 10 FEATURES MAIS IMPORTANTES:")
        print("-" * 40)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        # ===== 7. SALVAR TUDO =====
        logger.info("Salvando modelo e resultados")
        Path("data/models").mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo e preprocessador
        model_path = f"data/models/random_forest_{TASK_TYPE}.joblib"
        preprocessor_path = f"data/models/preprocessor_{TASK_TYPE}.joblib"
        
        model.save_model(model_path)
        preprocessor.save_preprocessors(preprocessor_path)
        
        # Salvar métricas
        import json
        metrics_data = {
            **{k: v for k, v in metrics.items() if k not in ['classification_report']},
            'cv_results': cv_results,
            'feature_importance': top_features.to_dict('records')
        }
        
        if TUNE_HYPERPARAMS:
            metrics_data['best_params'] = tuning_results['best_params']
        
        # Converter numpy arrays para listas
        for key, value in metrics_data.items():
            if isinstance(value, np.ndarray):
                metrics_data[key] = value.tolist()
        
        metrics_path = f"data/models/metrics_{TASK_TYPE}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   🤖 Modelo: {model_path}")
        print(f"   🔧 Preprocessador: {preprocessor_path}")
        print(f"   📊 Métricas: {metrics_path}")
        
        # ===== 8. INSTRUÇÕES FINAIS =====
        print("\n" + "="*50)
        print("🎉 CONVERSÃO CONCLUÍDA COM SUCESSO!")
        print("="*50)
        
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Execute o dashboard:")
        print("   python scripts/run_dashboard.py")
        print("\n2. Ou com Docker:")
        print("   docker-compose up --build")
        print("\n3. Acesse: http://localhost:8501")
        
        print("\n💡 DICAS:")
        print("• Use o modo 'Modelo Treinado' para ver todas as métricas")
        print("• Use o modo 'Predições' para fazer novas predições")
        print("• Seus arquivos estão salvos em data/models/")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro durante conversão: {str(e)}")
        print(f"\n❌ ERRO: {str(e)}")
        print("\n🔧 POSSÍVEIS SOLUÇÕES:")
        print("1. Verifique se o dataset está em data/raw/")
        print("2. Confirme o nome da coluna target")
        print("3. Verifique se não há valores incompatíveis nos dados")
        return False


if __name__ == "__main__":
    # Executar conversão
    success = main()
    
    if not success:
        print("\n📋 CHECKLIST DE PROBLEMAS COMUNS:")
        print("□ Dataset está em data/raw/?")
        print("□ Nome do arquivo está correto?")
        print("□ Coluna target existe?")
        print("□ Dados têm formato correto?")
        print("□ Dependências instaladas? (pip install -r requirements.txt)")
