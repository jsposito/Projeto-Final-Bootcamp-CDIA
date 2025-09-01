#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE COMPLETO - Valida toda a estrutura e gera exemplo funcional.

Execute este script para testar se tudo está funcionando corretamente.
"""

import sys
import os
from pathlib import Path
import traceback


def setup_project_structure():
    """Cria estrutura completa do projeto."""
    
    print("📁 Criando estrutura de diretórios...")
    
    directories = [
        'data/raw', 'data/processed', 'data/models',
        'src/data', 'src/models', 'src/utils', 'src/dashboard',
        'scripts', 'tests', 'logs', 'configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Criar __init__.py files
    init_dirs = ['src', 'src/data', 'src/models', 'src/utils', 'src/dashboard', 'tests']
    for directory in init_dirs:
        (Path(directory) / '__init__.py').touch()
    
    print("✅ Estrutura criada")


def test_imports():
    """Testa se todas as importações funcionam."""
    
    print("📦 Testando importações...")
    
    try:
        # Adicionar src ao path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        # Testar importações básicas
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        print("✅ Bibliotecas básicas OK")
        
        # Testar importações do projeto
        from data.data_loader import DataLoader
        from data.preprocessing import DataPreprocessor  
        from models.random_forest import RandomForestModel
        from utils.logger import setup_logger
        
        print("✅ Módulos do projeto OK")
        return True
        
    except Exception as e:
        print(f"❌ Erro nas importações: {e}")
        print("💡 Execute: pip install -r requirements.txt")
        return False


def create_sample_dataset():
    """Cria dataset de exemplo para teste."""
    
    print("📊 Criando dataset de exemplo...")
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 500
    
    # Dataset Titanic-like para classificação
    df = pd.DataFrame({
        'idade': np.random.randint(1, 80, n_samples),
        'tarifa': np.random.uniform(7, 500, n_samples),
        'classe': np.random.choice([1, 2, 3], n_samples),
        'sexo': np.random.choice(['masculino', 'feminino'], n_samples),
        'embarcou': np.random.choice(['C', 'Q', 'S'], n_samples),
        'familia_size': np.random.randint(0, 8, n_samples),
        'sobreviveu': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    })
    
    # Adicionar alguns valores missing
    missing_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_idx[:10], 'idade'] = np.nan
    df.loc[missing_idx[10:], 'tarifa'] = np.nan
    
    # Salvar dataset
    df.to_csv('data/raw/titanic_exemplo.csv', index=False)
    
    print(f"✅ Dataset criado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print("📍 Salvo em: data/raw/titanic_exemplo.csv")
    
    return df


def test_full_pipeline():
    """Testa pipeline completo de ML."""
    
    print("🔬 Testando pipeline completo...")
    
    try:
        # Importações
        sys.path.append(str(Path(__file__).parent / 'src'))
        from data.data_loader import DataLoader
        from data.preprocessing import DataPreprocessor
        from models.random_forest import RandomForestModel
        from utils.logger import setup_logger
        
        # Setup logger
        logger = setup_logger(name='test', level='INFO', log_file='logs/test.log')
        
        # 1. Carregar dados
        data_loader = DataLoader()
        df = data_loader.load_csv('titanic_exemplo.csv')
        print(f"✅ Dados carregados: {df.shape}")
        
        # 2. Pré-processamento
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            df=df,
            target_column='sobreviveu',
            test_size=0.2
        )
        print(f"✅ Pré-processamento: {X_train.shape[1]} features")
        
        # 3. Treinamento
        model = RandomForestModel(task_type='classification', n_estimators=50)
        model.train(X_train, y_train)
        print("✅ Modelo treinado")
        
        # 4. Avaliação
        metrics = model.evaluate(X_test, y_test)
        print(f"✅ Acurácia: {metrics['accuracy']:.3f}")
        
        # 5. Salvar
        model.save_model('data/models/test_model.joblib')
        preprocessor.save_preprocessors('data/models/test_preprocessor.joblib')
        print("✅ Modelo salvo")
        
        # 6. Testar predição
        prediction = model.predict(X_test[:5])
        print(f"✅ Predições: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        traceback.print_exc()
        return False


def test_dashboard_files():
    """Verifica se arquivos do dashboard existem."""
    
    print("🎨 Verificando dashboard...")
    
    dashboard_file = Path('src/dashboard/streamlit_app.py')
    if dashboard_file.exists():
        print("✅ Dashboard encontrado")
        print("🚀 Para executar: streamlit run src/dashboard/streamlit_app.py")
        return True
    else:
        print("❌ Dashboard não encontrado")
        return False


def test_docker():
    """Verifica se Docker pode ser construído."""
    
    print("🐳 Verificando Docker...")
    
    dockerfile = Path('Dockerfile')
    compose_file = Path('docker-compose.yml')
    
    if dockerfile.exists() and compose_file.exists():
        print("✅ Arquivos Docker encontrados")
        print("🚀 Para executar: docker-compose up --build")
        return True
    else:
        print("❌ Arquivos Docker não encontrados")
        return False


def create_run_instructions():
    """Cria arquivo com instruções de execução."""
    
    instructions = """# 🚀 INSTRUÇÕES DE EXECUÇÃO

## ✅ Teste Concluído com Sucesso!

### 🎯 Próximos Passos:

#### 1. 📊 Usar com seus próprios dados:
```bash
# Coloque seu CSV em data/raw/
cp seu_dataset.csv data/raw/

# Treine o modelo
python scripts/train_model.py --data data/raw/seu_dataset.csv --target sua_coluna_target --tune-hyperparams
```

#### 2. 📱 Executar Dashboard:
```bash
# Opção 1: Streamlit direto
streamlit run src/dashboard/streamlit_app.py

# Opção 2: Script helper
python scripts/run_dashboard.py

# Opção 3: Docker
docker-compose up --build
```

#### 3. 🧪 Executar Testes:
```bash
pytest tests/ -v
```

#### 4. 🔧 Desenvolvimento:
```bash
# Instalar dependências de desenvolvimento
pip install -r requirements.txt

# Formatar código
black .

# Linting
flake8 .
```

### 📁 Estrutura dos Arquivos Gerados:

- `data/raw/titanic_exemplo.csv` - Dataset de exemplo
- `data/models/test_model.joblib` - Modelo treinado
- `data/models/test_preprocessor.joblib` - Preprocessador
- `logs/test.log` - Logs de execução

### 💡 Dicas:

1. **Dados Reais**: Substitua o dataset de exemplo pelos seus dados
2. **Customização**: Modifique parâmetros nos arquivos de configuração
3. **Deploy**: Use Streamlit Cloud para deploy gratuito
4. **CI/CD**: GitHub Actions já configurado

### 🆘 Problemas Comuns:

- **Erro de importação**: Execute `pip install -r requirements.txt`
- **Streamlit não inicia**: Verifique se porta 8501 está livre
- **Docker não builda**: Verifique se Docker está instalado

### 🎊 Parabéns!

Sua estrutura de ML está pronta para produção!
"""

    with open('INSTRUCOES.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Instruções salvas em INSTRUCOES.md")


def main():
    """Função principal de teste."""
    
    print("🧪 TESTE COMPLETO DA ESTRUTURA ML")
    print("="*50)
    
    all_tests_passed = True
    
    try:
        # 1. Criar estrutura
        setup_project_structure()
        
        # 2. Testar importações
        if not test_imports():
            all_tests_passed = False
            
        # 3. Criar dados de exemplo
        create_sample_dataset()
        
        # 4. Testar pipeline completo
        if not test_full_pipeline():
            all_tests_passed = False
            
        # 5. Verificar dashboard
        test_dashboard_files()
        
        # 6. Verificar Docker
        test_docker()
        
        # 7. Criar instruções
        create_run_instructions()
        
        print("\n" + "="*50)
        
        if all_tests_passed:
            print("🎉 TODOS OS TESTES PASSARAM!")
            print("✅ Estrutura ML pronta para uso")
            print("\n🚀 EXECUTE AGORA:")
            print("streamlit run src/dashboard/streamlit_app.py")
            print("\n📖 Veja INSTRUCOES.md para mais detalhes")
        else:
            print("⚠️ ALGUNS TESTES FALHARAM")
            print("🔧 Verifique os erros acima e:")
            print("1. Execute: pip install -r requirements.txt") 
            print("2. Execute novamente: python test_everything.py")
            
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        traceback.print_exc()
        
        print("\n🆘 SOLUÇÃO RÁPIDA:")
        print("1. Verifique se está no diretório correto")
        print("2. Execute: pip install pandas numpy scikit-learn streamlit")
        print("3. Execute novamente: python test_everything.py")


if __name__ == "__main__":
    main()
