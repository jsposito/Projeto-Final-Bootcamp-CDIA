# Projeto-Final-Bootcamp-CDIA
Bootcamp Ciência de Dados e Inteligência Artificial
# Projeto de Manutenção Preditiva - Bootcamp CDIA

## 1. Objetivo do Projeto

Este projeto visa desenvolver um sistema inteligente de manutenção preditiva para uma empresa do setor industrial. O objetivo é criar um modelo de Machine Learning capaz de prever a probabilidade de 5 tipos diferentes de falhas em máquinas, utilizando dados de sensores IoT. [cite_start]A solução permite que a equipe de manutenção atue de forma proativa, reduzindo custos com paradas inesperadas e otimizando a produção. 

## 2. Descrição dos Dados

[cite_start]O conjunto de dados utilizado foi o `bootcamp_train.csv`, contendo informações sobre atributos de máquinas, como: 
- `tipo`: Tipo de produto/máquina (L, M, H)
- `temperatura_ar`: Temperatura do ar no ambiente (K)
- `velocidade_rotacional`: Velocidade da máquina (RPM)
- `torque`: Torque da máquina (Nm)
- `desgaste_da_ferramenta`: Tempo de uso da ferramenta (minutos)
- **Colunas-alvo**: 5 colunas binárias indicando tipos específicos de falha (FDF, FDC, FP, FTE, FA).

## 3. Como Executar o Projeto

1.  **Ambiente:** O código foi desenvolvido para ser executado em um ambiente Google Colab.
2.  **Arquivo:** Faça o upload do notebook `Preparação_completa_do_terreno (1).ipynb` para o Colab.
3.  **Execução:** Execute as células do notebook em ordem sequencial. Será solicitado o upload do arquivo `bootcamp_train.csv`.
4.  **Resultado:** Ao final, o notebook irá gerar e baixar automaticamente o arquivo `submission.csv` com as previsões de probabilidade para um conjunto de teste hipotético.

## 4. Decisões de Modelagem e Justificativas

-   **Tratamento de Dados:** A análise inicial revelou dados ausentes e inconsistentes. Os valores ausentes foram preenchidos com a **mediana** (robusta a outliers) e os rótulos de falha foram padronizados para um formato binário (0/1).
-   **Abordagem do Problema:** O problema foi modelado como uma **classificação multirrótulo**, utilizando o `MultiOutputClassifier` do Scikit-learn. Esta abordagem treina um classificador independente para cada tipo de falha, o que é ideal para o problema.
-   **Algoritmo:** Foi utilizado um `RandomForestClassifier` devido à sua alta performance com dados tabulares. O parâmetro `class_weight='balanced'` foi ativado para mitigar o forte **desbalanceamento de classes** identificado na análise exploratória.

## 5. Métricas de Avaliação Utilizadas

Dado o desbalanceamento das classes de falha, a acurácia não é uma métrica confiável. Por isso, utilizamos:
-   **Classification Report:** Fornece `Precision`, `Recall` e `F1-Score` para cada tipo de falha, oferecendo uma visão detalhada do desempenho do modelo em prever as classes minoritárias (as falhas).
-   **ROC AUC Score:** Mede a capacidade do modelo de distinguir corretamente entre classes positivas e negativas.
