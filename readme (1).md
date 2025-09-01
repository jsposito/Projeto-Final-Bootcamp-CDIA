# 🌲 Random Forest ML Project

Um projeto completo de Machine Learning usando Random Forest com infraestrutura profissional, dashboard interativo e containerização Docker.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)

## 🚀 Funcionalidades

- ✅ **Código Modular**: Organizado em scripts Python documentados
- ✅ **Dashboard Interativo**: Interface Streamlit para análise e predições
- ✅ **Containerização**: Pronto para Docker
- ✅ **CI/CD Pipeline**: Automação com GitHub Actions
- ✅ **Pré-processamento Avançado**: Pipeline completo de preparação de dados
- ✅ **Hiperparâmetros**: Busca automática dos melhores parâmetros
- ✅ **Validação Cruzada**: Avaliação robusta do modelo
- ✅ **Logging**: Sistema completo de logs
- ✅ **Métricas Detalhadas**: Avaliação completa de performance

## 📁 Estrutura do Projeto

```
📦 projeto-ml/
├── 📄 README.md                    # Este arquivo
├── 📄 requirements.txt             # Dependências Python
├── 🐳 Dockerfile                   # Container Docker
├── 📄 .gitignore                   # Arquivos ignorados
├── 📄 docker-compose.yml           # Orquestração Docker
│
├── 🔧 .github/
│   └── workflows/
│       └── ci.yml                  # Pipeline CI/CD
│
├── 💻 src/                         # Código fonte principal
│   ├── 📊 data/
│   │   ├── data_loader.py          # Carregamento de dados
│   │   └── preprocessing.py        # Pré-processamento
│   │
│   ├── 🤖 models/
│   │   ├── random_forest.py        # Modelo Random Forest
│   │   ├── trainer.py              # Treinamento
│   │   └── evaluator.py            # Avaliação
│   │
│   ├── 🔧 utils/
│   │   ├── logger.py               # Sistema de logs
│   │   └── config.py               # Configurações
│   │
│   └── 📱 dashboard/
│       └── streamlit_app.py        # Dashboard Streamlit
│
├── 🎯 scripts/                     # Scripts executáveis
│   ├── train_model.py              # Script de treinamento
│   ├── evaluate_model.py           # Script de avaliação
│   └── run_dashboard.py            # Executar dashboard
│
├── 📊 data/                        # Dados
│   ├── raw/                        # Dados brutos
│   ├── processed/                  # Dados processados
│   └── models/                     # Modelos salvos
│
├── 🧪 tests/                       # Testes automatizados
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
└── ⚙️ configs/                     # Configurações
    ├── model_config.yaml
    └── app_config.yaml
```

## 🏃‍♂️ Quick Start

### 1. Instalação

```bash
# Clonar repositório
git clone <seu-repositorio>
cd projeto-ml

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar Dados

```bash
# Colocar seu dataset em data/raw/
cp seu_dataset.csv data/raw/
```

### 3. Treinar Modelo

```bash
# Treinamento básico
python scripts/train_model.py --data data/raw/seu_dataset.csv --target sua_coluna_target

# Com busca de hiperparâmetros
python scripts/train_model.py --data data/raw/seu_dataset.csv --target sua_coluna_target --tune-hyperparams

# Para regressão
python scripts/train_model.py --data data/raw/seu_dataset.csv --target sua_coluna_target --task regression
```

### 4. Executar Dashboard

```bash
# Localmente
streamlit run src/dashboard/streamlit_app.py

# Ou usar o script helper
python scripts/run_dashboard.py
```

### 5. Docker (Recomendado)

```bash
# Construir imagem
docker build -t ml-project .

# Executar container
docker run -p 8501:8501 ml-project

# Acessar: http://localhost:8501
```

## 🎯 Uso Detalhado

### Treinamento do Modelo

O script `train_model.py` oferece várias opções:

```bash
python scripts/train_model.py \
    --data data/raw/titanic.csv \
    --target survived \
    --task classification \
    --test-size 0.2 \
    --tune-hyperparams \
    --output-dir data/models \
    --log-level INFO
```

**Parâmetros disponíveis:**
- `--data`: Caminho para o dataset CSV
- `--target`: Nome da coluna target
- `--task`: `classification` ou `regression`
- `--test-size`: Proporção para teste (padrão: 0.2)
- `--tune-hyperparams`: Ativa busca de hiperparâmetros
- `--output-dir`: Diretório para salvar modelo
- `--log-level`: Nível de logging

### Dashboard Streamlit

O dashboard oferece três modos:

1. **📊 Análise Exploratória**
   - Upload de CSV
   - Estatísticas descritivas
   - Visualizações automáticas
   - Análise de correlação

2. **🤖 Modelo Treinado**
   - Métricas de performance
   - Importância das features
   - Matriz de confusão
   - Resultados de validação cruzada

3. **🔮 Predições**
   - Interface para predições individuais
   - Predições em lote via upload
   - Download dos resultados

## 🔧 Configuração Avançada

### Variáveis de Ambiente

Crie um arquivo `.env`:

```env
# Configurações do modelo
MODEL_RANDOM_STATE=42
MODEL_N_ESTIMATORS=100

# Configurações de logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Configurações do Streamlit
STREAMLIT_PORT=8501
```

### Docker Compose

Para ambiente completo com volumes:

```yaml
version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Testes com cobertura
pytest tests/ --cov=src --cov-report=html

# Testes específicos
pytest tests/test_models.py -v
```

## 📊 Métricas e Avaliação

O projeto gera automaticamente:

- **Métricas de Classificação**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Métricas de Regressão**: R², RMSE, MAE, MSE
- **Validação Cruzada**: Score médio e desvio padrão
- **Feature Importance**: Ranking das variáveis mais importantes
- **Matriz de Confusão**: Para problemas de classificação

## 🐳 Deploy com Docker

### Build e Run

```bash
# Build da imagem
docker build -t ml-project:latest .

# Run do container
docker run -d \
    --name ml-container \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    ml-project:latest

# Ver logs
docker logs ml-container

# Parar container
docker stop ml-container
```

### Deploy em Produção

Para deploy gratuito, considere:

- **Streamlit Cloud**: Deploy direto do GitHub
- **Heroku**: Container deployment gratuito
- **Railway**: Deploy com Docker gratuito

## 📈 Melhorias Futuras

- [ ] API REST com FastAPI
- [ ] Monitoramento de modelo drift
- [ ] Integração com MLflow
- [ ] Testes A/B para modelos
- [ ] Deploy automatizado
- [ ] Notificações de performance

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

Seu Nome - [GitHub](https://github.com/seu-usuario)

---

**💡 Dica**: Este projeto foi estruturado seguindo as melhores práticas de MLOps e pode ser facilmente adaptado para diferentes problemas de Machine Learning!