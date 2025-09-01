# -*- coding: utf-8 -*-
"""
Módulo para pré-processamento de dados.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pré-processamento de dados."""
    
    def __init__(self):
        """Inicializa o pré-processador."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Trata valores missing.
        
        Args:
            df: DataFrame de entrada
            strategy: Estratégia para imputação ('mean', 'median', 'most_frequent', 'constant')
            
        Returns:
            DataFrame sem valores missing
        """
        df_processed = df.copy()
        
        # Separar colunas numéricas e categóricas
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Imputar valores numéricos
        if len(numeric_cols) > 0:
            if 'numeric' not in self.imputers:
                self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = self.imputers['numeric'].fit_transform(df_processed[numeric_cols])
            else:
                df_processed[numeric_cols] = self.imputers['numeric'].transform(df_processed[numeric_cols])
                
        # Imputar valores categóricos
        if len(categorical_cols) > 0:
            if 'categorical' not in self.imputers:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = self.imputers['categorical'].fit_transform(df_processed[categorical_cols])
            else:
                df_processed[categorical_cols] = self.imputers['categorical'].transform(df_processed[categorical_cols])
                
        logger.info(f"Valores missing tratados usando estratégia: {strategy}")
        return df_processed
        
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: Optional[List[str]] = None,
                                  encoding_type: str = 'onehot') -> pd.DataFrame:
        """
        Codifica variáveis categóricas.
        
        Args:
            df: DataFrame de entrada
            categorical_columns: Lista de colunas categóricas (None para auto-detect)
            encoding_type: Tipo de encoding ('onehot' ou 'label')
            
        Returns:
            DataFrame com variáveis codificadas
        """
        df_processed = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            
        if not categorical_columns:
            logger.info("Nenhuma variável categórica encontrada")
            return df_processed
            
        for col in categorical_columns:
            if col in df_processed.columns:
                if encoding_type == 'onehot':
                    # One-hot encoding
                    if col not in self.encoders:
                        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                        encoded_data = encoder.fit_transform(df_processed[[col]])
                        self.encoders[col] = encoder
                    else:
                        encoded_data = self.encoders[col].transform(df_processed[[col]])
                        
                    # Criar nomes das colunas
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_processed.index)
                    
                    # Substituir coluna original
                    df_processed = df_processed.drop(columns=[col])
                    df_processed = pd.concat([df_processed, encoded_df], axis=1)
                    
                elif encoding_type == 'label':
                    # Label encoding
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
                    else:
                        df_processed[col] = self.encoders[col].transform(df_processed[col])
                        
        logger.info(f"Codificação {encoding_type} aplicada em {len(categorical_columns)} colunas")
        return df_processed
        
    def scale_features(self, df: pd.DataFrame, 
                      columns_to_scale: Optional[List[str]] = None,
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Escala features numéricas.
        
        Args:
            df: DataFrame de entrada
            columns_to_scale: Colunas para escalar (None para todas numéricas)
            scaler_type: Tipo de scaler ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame com features escaladas
        """
        df_processed = df.copy()
        
        if columns_to_scale is None:
            columns_to_scale = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
        if not columns_to_scale:
            logger.info("Nenhuma coluna numérica para escalar")
            return df_processed
            
        if scaler_type not in self.scalers:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            # Adicionar outros scalers conforme necessário
            
            df_processed[columns_to_scale] = self.scalers[scaler_type].fit_transform(
                df_processed[columns_to_scale]
            )
        else:
            df_processed[columns_to_scale] = self.scalers[scaler_type].transform(
                df_processed[columns_to_scale]
            )
            
        logger.info(f"Scaling {scaler_type} aplicado em {len(columns_to_scale)} colunas")
        return df_processed
        
    def split_data(self, df: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino e teste.
        
        Args:
            df: DataFrame completo
            target_column: Nome da coluna target
            test_size: Proporção para teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if target_column not in df.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        
        logger.info(f"Dados divididos: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        return X_train, X_test, y_train, y_test
        
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str,
                          missing_strategy: str = 'mean',
                          encoding_type: str = 'onehot',
                          scale_features: bool = True,
                          test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            df: DataFrame original
            target_column: Nome da coluna target
            missing_strategy: Estratégia para valores missing
            encoding_type: Tipo de encoding para categóricas
            scale_features: Se deve escalar features
            test_size: Proporção para teste
            
        Returns:
            X_train, X_test, y_train, y_test processados
        """
        logger.info("Iniciando pipeline de pré-processamento")
        
        # 1. Tratar valores missing
        df_processed = self.handle_missing_values(df, missing_strategy)
        
        # 2. Separar features do target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # 3. Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if y.dtype == 'object' else None
        )
        
        # 4. Codificar variáveis categóricas
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            X_train = self.encode_categorical_features(X_train, categorical_cols, encoding_type)
            X_test = self.encode_categorical_features(X_test, categorical_cols, encoding_type)
            
        # 5. Escalar features (opcional)
        if scale_features:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X_train = self.scale_features(X_train, numeric_cols)
                X_test = self.scale_features(X_test, numeric_cols)
                
        # Salvar nomes das features
        self.feature_names = X_train.columns.tolist()
        
        logger.info("Pipeline de pré-processamento concluído")
        logger.info(f"Features finais: {len(self.feature_names)} colunas")
        
        return X_train, X_test, y_train, y_test
        
    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features após processamento."""
        return self.feature_names.copy()
        
    def save_preprocessors(self, filepath: str) -> None:
        """
        Salva os pré-processadores treinados.
        
        Args:
            filepath: Caminho para salvar
        """
        import joblib
        
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessors, filepath)
        logger.info(f"Pré-processadores salvos em: {filepath}")
        
    def load_preprocessors(self, filepath: str) -> None:
        """
        Carrega pré-processadores salvos.
        
        Args:
            filepath: Caminho do arquivo
        """
        import joblib
        
        preprocessors = joblib.load(filepath)
        self.scalers = preprocessors['scalers']
        self.encoders = preprocessors['encoders']
        self.imputers = preprocessors['imputers']
        self.feature_names = preprocessors['feature_names']
        
        logger.info(f"Pré-processadores carregados de: {filepath}")