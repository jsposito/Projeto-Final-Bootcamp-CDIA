# -*- coding: utf-8 -*-
"""
Módulo para carregamento e validação de dados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Classe responsável pelo carregamento e validação de dados."""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Inicializa o carregador de dados.
        
        Args:
            data_path: Caminho para os dados brutos
        """
        self.data_path = Path(data_path)
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Carrega arquivo CSV com validação básica.
        
        Args:
            filename: Nome do arquivo CSV
            **kwargs: Argumentos adicionais para pd.read_csv
            
        Returns:
            DataFrame carregado
            
        Raises:
            FileNotFoundError: Se arquivo não existir
            ValueError: Se arquivo estiver vazio ou corrompido
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
            if df.empty:
                raise ValueError("Dataset está vazio")
                
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar {filename}: {str(e)}")
            raise
            
    def load_multiple_files(self, filenames: list) -> dict:
        """
        Carrega múltiplos arquivos CSV.
        
        Args:
            filenames: Lista de nomes de arquivos
            
        Returns:
            Dicionário com DataFrames carregados
        """
        datasets = {}
        
        for filename in filenames:
            try:
                datasets[filename] = self.load_csv(filename)
                logger.info(f"✓ {filename} carregado com sucesso")
            except Exception as e:
                logger.warning(f"✗ Falha ao carregar {filename}: {str(e)}")
                
        return datasets
        
    def validate_dataset(self, df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
        """
        Valida estrutura básica do dataset.
        
        Args:
            df: DataFrame para validar
            required_columns: Colunas obrigatórias
            
        Returns:
            True se válido, False caso contrário
        """
        if df.empty:
            logger.error("Dataset está vazio")
            return False
            
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.error(f"Colunas obrigatórias ausentes: {missing_cols}")
                return False
                
        logger.info("Dataset validado com sucesso")
        return True
        
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Retorna informações básicas do dataset.
        
        Args:
            df: DataFrame para analisar
            
        Returns:
            Dicionário com informações do dataset
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        logger.info(f"Dataset info gerada: {info['shape'][0]} linhas, {info['shape'][1]} colunas")
        return info