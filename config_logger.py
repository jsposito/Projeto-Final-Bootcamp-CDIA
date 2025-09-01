# -*- coding: utf-8 -*-
"""
Sistema de logging configurável.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = __name__, 
                level: str = 'INFO',
                log_file: str = None,
                console_output: bool = True) -> logging.Logger:
    """
    Configura sistema de logging.
    
    Args:
        name: Nome do logger
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Arquivo para salvar logs (opcional)
        console_output: Se deve imprimir no console
        
    Returns:
        Logger configurado
    """
    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Evitar duplicação de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Formato de log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler para arquivo
    if log_file:
        # Criar diretório se não existir
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna logger existente ou cria um novo.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger
    """
    return logging.getLogger(name)