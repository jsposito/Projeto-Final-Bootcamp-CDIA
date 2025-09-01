#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para treinamento do modelo Random Forest.

Uso:
    python scripts/train_model.py --data data/raw/dataset.csv --target target_column
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel
from utils.logger import setup_logger
import pandas as pd


def main():
    """Função principal do script de treinamento."""
    
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Treinar modelo Random Forest')
    parser.add_argument('--data', type=str, required=True,
                       help='Caminho para o arquivo de dados')
    parser.add_argument('--target', type=str, required=True,
                       help='Nome da coluna target')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Tipo de tarefa')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporção dos dados para teste')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Realizar busca de hiperparâmetros')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Diretório para salvar modelo')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nível de log')
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logger(
        name='train_model',
        level=args.log_level,
        log_file=f'logs/training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    try:
        logger.info("="*50)
        logger.info("INICIANDO TREINAMENTO DO MODELO")
        logger.info("="*50)
        
        # 1. Carregar dados
        logger.info(f"Carregando dados de: {args.data}")
        data_loader = DataLoader()
        df = data_loader.load_csv(Path(args.data).name)
        
        # Validar se target existe
        if args.target not in df.columns:
            raise ValueError(f"Coluna target '{args.target}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        
        # 2. Pré-processamento
        logger.info("Iniciando pré-processamento")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column=args.target,
            test_size=args.test_size
        )
        
        logger.info(f"Dados processados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
        # 3. Criar e treinar modelo
        logger.info(f"Criando modelo Random Forest para {args.task}")
        model = RandomForestModel(task_type=args.task)
        
        if args.tune_hyperparams:
            logger.info("Realizando busca de hiperparâmetros")
            tuning_results = model.hyperparameter_tuning(X_train, y_train)
            logger.info(f"Melhores parâmetros: {tuning_results['best_params']}")
        else:
            model.train(X_train, y_train)
            
        # 4. Avaliar modelo
        logger.info("Avaliando modelo")
        metrics = model.evaluate(X_test, y_test)
        
        # Log das métricas
        logger.info("RESULTADOS DA AVALIAÇÃO:")
        for metric, value in metrics.items():
            if metric not in ['classification_report', 'confusion_matrix']:
                logger.info(f"{metric}: {value}")
                
        # 5. Validação cruzada
        cv_results = model.cross_validate(X_train, y_train)
        logger.info(f"Validação cruzada: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
        
        # 6. Feature importance
        logger.info("TOP 10 FEATURES MAIS IMPORTANTES:")
        top_features = model.get_feature_importance(top_n=10)
        for _, row in top_features.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
        # 7. Salvar modelo e preprocessadores
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"random_forest_{args.task}.joblib"
        preprocessor_path = output_dir / f"preprocessor_{args.task}.joblib"
        
        model.save_model(str(model_path))
        preprocessor.save_preprocessors(str(preprocessor_path))
        
        # 8. Salvar métricas
        metrics_path = output_dir / f"metrics_{args.task}.json"
        import json
        
        # Preparar métricas para JSON
        metrics_json = {}
        for k, v in metrics.items():
            if k not in ['classification_report']:
                if isinstance(v, (int, float, str, list)):
                    metrics_json[k] = v
                else:
                    metrics_json[k] = str(v)
                    
        metrics_json['cv_results'] = cv_results
        metrics_json['feature_importance'] = top_features.to_dict('records')
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Métricas salvas em: {metrics_path}")
        
        logger.info("="*50)
        logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("="*50)
        logger.info(f"Modelo salvo em: {model_path}")
        logger.info(f"Preprocessadores salvos em: {preprocessor_path}")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()