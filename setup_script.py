#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de setup automático para o projeto.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Cria estrutura de diretórios necessária."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'logs',
        'tests',
        'configs',
        'src/data',
        'src/models',
        'src/utils',
        'src/dashboard',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Criado: {directory}/")


def create_init_files():
    """Cria arquivos __init__.py necessários."""
    init_dirs = [
        'src',
        'src/data',
        'src/models', 
        'src/utils',
        'src/dashboard',
        'tests'
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        init_file.touch()
        print(f"✅ Criado: {init_file}")


def create_config_files():
    """Cria arquivos de configuração básicos."""
    
    # model_config.yaml
    model_config = """# Configurações do modelo Random Forest
random_forest:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
  n_jobs: -1

# Configurações de pré-processamento  
preprocessing:
  missing_strategy: 'mean'
  encoding_type: 'onehot'
  scale_features: true
  test_size: 0.2

# Configurações de busca de hiperparâmetros
hyperparameter_tuning:
  cv_folds: 5
  param_grid:
    n_estimators: [50, 100, 200]
    max_depth: [null, 10, 20, 30]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
"""
    
    with open('configs/model_config.yaml', 'w', encoding='utf-8') as f:
        f.write(model_config)
    print("✅ Criado: configs/model_config.yaml")
    
    # app_config.yaml
    app_config = """# Configurações da aplicação
app:
  name: "Random Forest ML Project"
  version: "1.0.0"
  description: "Projeto de ML com Random Forest"

# Configurações do Streamlit
streamlit:
  title: "Random Forest Dashboard"
  port: 8501
  theme: "dark"

# Configurações de logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/app.log"
"""
    
    with open('configs/app_config.yaml', 'w', encoding='utf-8') as f:
        f.write(app_config)
    print("✅ Criado: configs/app_config.yaml")


def create_sample_test():
    """Cria teste de exemplo."""
    test_content = """# -*- coding: utf-8 -*-
\"\"\"
Testes básicos para o projeto.
\"\"\"

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel


def test_data_loader():
    \"\"\"Teste básico do carregador de dados.\"\"\"
    loader = DataLoader()
    assert loader is not None


def test_preprocessor():
    \"\"\"Teste básico do preprocessador.\"\"\"
    preprocessor = DataPreprocessor()
    assert preprocessor is not None
    
    # Teste com dados fictícios
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    })
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, 'target', test_size=0.2
    )
    
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_random_forest_model():
    \"\"\"Teste básico do modelo Random Forest.\"\"\"
    model = RandomForestModel(task_type='classification')
    assert model is not None
    assert model.task_type == 'classification'
    
    # Teste com dados fictícios
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model.train(X, y)
    assert model.is_trained
    
    predictions = model.predict(X[:10])
    assert len(predictions) == 10


if __name__ == "__main__":
    pytest.main([__file__])
"""
    
    with open('tests/test_basic.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    print("✅ Criado: tests/test_basic.py")


def main():
    """Função principal do setup."""
    print("🚀 Configurando projeto Random Forest ML...")
    print("=" * 50)
    
    try:
        # Criar estrutura de diretórios
        print("\n📁 Criando diretórios...")
        create_directories()
        
        # Criar arquivos __init__.py
        print("\n📄 Criando arquivos __init__.py...")
        create_init_files()
        
        # Criar arquivos de configuração
        print("\n⚙️ Criando arquivos de configuração...")
        create_config_files()
        
        # Criar teste de exemplo
        print("\n🧪 Criando testes de exemplo...")
        create_sample_test()
        
        print("\n" + "=" * 50)
        print("✅ Setup concluído com sucesso!")
        print("\n📋 Próximos passos:")
        print("1. pip install -r requirements.txt")
        print("2. Coloque seus dados em data/raw/")
        print("3. python scripts/train_model.py --data data/raw/seu_dataset.csv --target sua_coluna")
        print("4. python scripts/run_dashboard.py")
        print("\n🐳 Para usar Docker:")
        print("docker-compose up --build")
        
    except Exception as e:
        print(f"❌ Erro durante setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()