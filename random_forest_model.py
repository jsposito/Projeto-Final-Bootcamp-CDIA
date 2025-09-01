# -*- coding: utf-8 -*-
"""
Modelo Random Forest com funcionalidades avançadas.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Classe para modelo Random Forest com funcionalidades completas."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """
        Inicializa o modelo Random Forest.
        
        Args:
            task_type: 'classification' ou 'regression'
            **kwargs: Parâmetros para o modelo
        """
        self.task_type = task_type
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.best_params = None
        
        # Parâmetros padrão
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        # Criar modelo baseado no tipo de task
        if task_type == 'classification':
            self.model = RandomForestClassifier(**default_params)
        elif task_type == 'regression':
            self.model = RandomForestRegressor(**default_params)
        else:
            raise ValueError("task_type deve ser 'classification' ou 'regression'")
            
        logger.info(f"Modelo Random Forest ({task_type}) inicializado")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Treina o modelo.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        """
        logger.info("Iniciando treinamento do modelo")
        
        # Validar dados
        if X_train.empty or y_train.empty:
            raise ValueError("Dados de treino não podem estar vazios")
            
        # Treinar modelo
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calcular importância das features
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Treinamento concluído com sucesso")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com predições
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
            
        predictions = self.model.predict(X)
        logger.info(f"Predições realizadas para {X.shape[0]} amostras")
        
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições de probabilidade (apenas para classificação).
        
        Args:
            X: Features para predição
            
        Returns:
            Array com probabilidades
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba disponível apenas para classificação")
            
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
            
        probabilities = self.model.predict_proba(X)
        logger.info(f"Probabilidades calculadas para {X.shape[0]} amostras")
        
        return probabilities
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia o modelo.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes da avaliação")
            
        predictions = self.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = self._evaluate_classification(y_test, predictions, X_test)
        else:
            metrics = self._evaluate_regression(y_test, predictions)
            
        logger.info("Avaliação do modelo concluída")
        return metrics
        
    def _evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Avalia modelo de classificação."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Adicionar probabilidades se disponível
        if hasattr(self.model, 'predict_proba'):
            try:
                probas = self.predict_proba(X_test)
                if probas.shape[1] == 2:  # Classificação binária
                    from sklearn.metrics import roc_auc_score
                    metrics['roc_auc'] = roc_auc_score(y_true, probas[:, 1])
            except Exception as e:
                logger.warning(f"Não foi possível calcular ROC-AUC: {e}")
                
        return metrics
        
    def _evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Avalia modelo de regressão."""
        from sklearn.metrics import mean_absolute_error
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
        
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            param_grid: Optional[Dict] = None,
                            cv: int = 5) -> Dict[str, Any]:
        """
        Realiza busca de hiperparâmetros.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            param_grid: Grid de parâmetros (None para usar padrão)
            cv: Número de folds para cross-validation
            
        Returns:
            Melhores parâmetros encontrados
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        logger.info("Iniciando busca de hiperparâmetros")
        
        # Configurar scoring baseado no tipo de task
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        
        # Atualizar importância das features
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Melhores parâmetros: {self.best_params}")
        logger.info(f"Melhor score CV: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Realiza validação cruzada.
        
        Args:
            X: Features
            y: Target
            cv: Número de folds
            
        Returns:
            Métricas de validação cruzada
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes da validação cruzada")
            
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        cv_metrics = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"Validação cruzada: {cv_metrics['mean_score']:.4f} (+/- {cv_metrics['std_score']*2:.4f})")
        
        return cv_metrics
        
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Retorna importância das features.
        
        Args:
            top_n: Número de features mais importantes (None para todas)
            
        Returns:
            DataFrame com importância das features
        """
        if self.feature_importance is None:
            raise ValueError("Modelo deve ser treinado primeiro")
            
        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance
        
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de ser salvo")
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'task_type': self.task_type,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.task_type = model_data['task_type']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data.get('best_params')
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Modelo carregado de: {filepath}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações do modelo.
        
        Returns:
            Dicionário com informações do modelo
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
            
        info = {
            'status': 'trained',
            'task_type': self.task_type,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_importance) if self.feature_importance is not None else 0,
            'best_params': self.best_params
        }
        
        return info