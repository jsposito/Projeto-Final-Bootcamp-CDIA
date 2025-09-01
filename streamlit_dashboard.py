# -*- coding: utf-8 -*-
"""
Dashboard Streamlit para análise e predições do modelo Random Forest.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from models.random_forest import RandomForestModel


# Configuração da página
st.set_page_config(
    page_title="Random Forest ML Dashboard",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """Carrega dataset com cache."""
    try:
        data_loader = DataLoader()
        return data_loader.load_csv(Path(file_path).name)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_trained_model(model_path: str, preprocessor_path: str):
    """Carrega modelo e preprocessador treinados."""
    try:
        model = RandomForestModel()
        model.load_model(model_path)
        
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessors(preprocessor_path)
        
        return model, preprocessor
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None


def load_metrics(metrics_path: str) -> dict:
    """Carrega métricas salvas."""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {e}")
        return {}


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 15):
    """Cria gráfico de importância das features."""
    top_features = feature_importance.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Features Mais Importantes',
        labels={'importance': 'Importância', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_confusion_matrix(cm: list, class_names: list = None):
    """Cria heatmap da matriz de confusão."""
    cm_array = np.array(cm)
    
    if class_names is None:
        class_names = [f'Classe {i}' for i in range(len(cm_array))]
    
    fig = px.imshow(
        cm_array,
        text_auto=True,
        aspect="auto",
        title="Matriz de Confusão",
        labels=dict(x="Predito", y="Real"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues'
    )
    
    return fig


def create_prediction_interface(model, preprocessor, feature_names: list):
    """Cria interface para fazer predições."""
    st.subheader("🔮 Fazer Predições")
    
    # Criar inputs para cada feature
    inputs = {}
    
    # Dividir em colunas para melhor layout
    n_cols = 3
    cols = st.columns(n_cols)
    
    for i, feature in enumerate(feature_names[:15]):  # Limitar a 15 features para UI
        col_idx = i % n_cols
        with cols[col_idx]:
            # Determinar tipo de input baseado no nome da feature
            if 'age' in feature.lower() or 'year' in feature.lower():
                inputs[feature] = st.number_input(f"{feature}", min_value=0, max_value=100, value=30)
            elif any(word in feature.lower() for word in ['price', 'cost', 'amount', 'salary']):
                inputs[feature] = st.number_input(f"{feature}", min_value=0.0, value=1000.0)
            elif feature.lower() in ['sex', 'gender']:
                inputs[feature] = st.selectbox(f"{feature}", ['Male', 'Female'])
            else:
                inputs[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("🚀 Fazer Predição", type="primary"):
        try:
            # Criar DataFrame com os inputs
            input_df = pd.DataFrame([inputs])
            
            # Fazer predição
            prediction = model.predict(input_df)
            
            # Mostrar resultado
            st.success(f"**Predição:** {prediction[0]}")
            
            # Se for classificação, mostrar probabilidades
            if model.task_type == 'classification' and hasattr(model.model, 'predict_proba'):
                probas = model.predict_proba(input_df)
                
                st.subheader("Probabilidades por Classe")
                for i, proba in enumerate(probas[0]):
                    st.metric(f"Classe {i}", f"{proba:.2%}")
                    
        except Exception as e:
            st.error(f"Erro na predição: {e}")


def main():
    """Função principal do dashboard."""
    
    # Título principal
    st.title("🌲 Random Forest ML Dashboard")
    st.markdown("### Análise e Predições com Machine Learning")
    
    # Sidebar para configurações
    st.sidebar.title("⚙️ Configurações")
    
    # Seleção de modo
    mode = st.sidebar.selectbox(
        "Selecione o modo:",
        ["📊 Análise Exploratória", "🤖 Modelo Treinado", "🔮 Predições"]
    )
    
    if mode == "📊 Análise Exploratória":
        st.header("📊 Análise Exploratória dos Dados")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Faça upload do dataset (CSV)",
            type=['csv'],
            help="Selecione um arquivo CSV para análise"
        )
        
        if uploaded_file is not None:
            # Carregar dados
            df = pd.read_csv(uploaded_file)
            
            # Informações básicas
            st.subheader("ℹ️ Informações do Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Linhas", df.shape[0])
            with col2:
                st.metric("Colunas", df.shape[1])
            with col3:
                st.metric("Valores Missing", df.isnull().sum().sum())
            with col4:
                st.metric("Memória (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
            
            # Mostrar primeiras linhas
            st.subheader("👀 Primeiras Linhas")
            st.dataframe(df.head())
            
            # Estatísticas descritivas
            st.subheader("📈 Estatísticas Descritivas")
            st.dataframe(df.describe())
            
            # Valores missing por coluna
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if not missing_data.empty:
                st.subheader("❌ Valores Missing por Coluna")
                fig_missing = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Valores Missing por Coluna",
                    labels={'x': 'Quantidade', 'y': 'Coluna'}
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            
            # Correlação (apenas para colunas numéricas)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("🔗 Matriz de Correlação")
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlação",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    
    elif mode == "🤖 Modelo Treinado":
        st.header("🤖 Análise do Modelo Treinado")
        
        # Verificar se arquivos de modelo existem
        model_files = list(Path("data/models").glob("*.joblib")) if Path("data/models").exists() else []
        metrics_files = list(Path("data/models").glob("*.json")) if Path("data/models").exists() else []
        
        if not model_files:
            st.warning("⚠️ Nenhum modelo treinado encontrado. Execute o script de treinamento primeiro.")
            st.code("python scripts/train_model.py --data seu_dataset.csv --target sua_coluna_target")
            return
        
        # Seleção de modelo
        selected_model = st.selectbox(
            "Selecione o modelo:",
            model_files,
            format_func=lambda x: x.name
        )
        
        # Carregar métricas se disponíveis
        metrics_file = None
        for f in metrics_files:
            if selected_model.stem in f.name:
                metrics_file = f
                break
        
        if metrics_file:
            metrics = load_metrics(str(metrics_file))
            
            # Mostrar métricas principais
            st.subheader("📊 Métricas do Modelo")
            
            if 'accuracy' in metrics:
                # Classificação
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Acurácia", f"{metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precisão", f"{metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
                    
                # Matriz de confusão
                if 'confusion_matrix' in metrics:
                    st.subheader("🎯 Matriz de Confusão")
                    fig_cm = plot_confusion_matrix(metrics['confusion_matrix'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            elif 'r2' in metrics:
                # Regressão
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                with col2:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                with col3:
                    st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                with col4:
                    st.metric("MSE", f"{metrics.get('mse', 0):.3f}")
            
            # Validação cruzada
            if 'cv_results' in metrics:
                cv_data = metrics['cv_results']
                st.subheader("✅ Validação Cruzada")
                st.metric(
                    "Score CV", 
                    f"{cv_data['mean_score']:.4f} ± {cv_data['std_score']*2:.4f}"
                )
            
            # Feature importance
            if 'feature_importance' in metrics:
                st.subheader("🎯 Importância das Features")
                
                importance_df = pd.DataFrame(metrics['feature_importance'])
                fig_importance = plot_feature_importance(importance_df)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Tabela de importância
                with st.expander("📋 Tabela de Importância das Features"):
                    st.dataframe(importance_df)
    
    elif mode == "🔮 Predições":
        st.header("🔮 Interface de Predições")
        
        # Verificar se modelo existe
        model_files = list(Path("data/models").glob("*random_forest*.joblib")) if Path("data/models").exists() else []
        preprocessor_files = list(Path("data/models").glob("*preprocessor*.joblib")) if Path("data/models").exists() else []
        
        if not model_files or not preprocessor_files:
            st.warning("⚠️ Modelo ou preprocessador não encontrado. Treine o modelo primeiro.")
            return
        
        try:
            # Carregar modelo e preprocessador
            model, preprocessor = load_trained_model(
                str(model_files[0]), 
                str(preprocessor_files[0])
            )
            
            if model and preprocessor:
                st.success("✅ Modelo carregado com sucesso!")
                
                # Interface de predição
                create_prediction_interface(model, preprocessor, preprocessor.get_feature_names())
                
                # Upload para predições em lote
                st.subheader("📊 Predições em Lote")
                batch_file = st.file_uploader(
                    "Upload arquivo CSV para predições em lote",
                    type=['csv'],
                    key="batch_prediction"
                )
                
                if batch_file is not None:
                    batch_df = pd.read_csv(batch_file)
                    st.dataframe(batch_df.head())
                    
                    if st.button("🚀 Executar Predições em Lote"):
                        try:
                            # Processar dados
                            processed_data = preprocessor.handle_missing_values(batch_df)
                            
                            # Fazer predições
                            predictions = model.predict(processed_data)
                            
                            # Adicionar predições ao DataFrame
                            result_df = batch_df.copy()
                            result_df['Predição'] = predictions
                            
                            st.success(f"✅ {len(predictions)} predições realizadas!")
                            st.dataframe(result_df)
                            
                            # Download dos resultados
                            csv = result_df.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="📥 Download Resultados",
                                data=csv,
                                file_name=f"predicoes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Erro nas predições em lote: {e}")
                            
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")


def main():
    """Função principal do dashboard."""
    
    # Sidebar com informações
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Estrutura do Projeto")
    st.sidebar.markdown("""
    - `data/raw/` - Dados originais
    - `data/models/` - Modelos treinados
    - `scripts/` - Scripts de treinamento
    - `src/` - Código fonte
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Como usar")
    st.sidebar.markdown("""
    1. **Análise Exploratória**: Upload seu CSV
    2. **Treinar Modelo**: Use o script de treino
    3. **Fazer Predições**: Use o modelo treinado
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Dica**: Treine seu modelo primeiro usando o script `train_model.py`")


if __name__ == "__main__":
    main()