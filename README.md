# Previsão de Cancelamento de Clientes (Churn) em Telecom com Machine Learning Clássico

Este projeto tem como objetivo **prever o cancelamento de clientes (churn)** em uma empresa de telecomunicações utilizando **modelos de Machine Learning clássico**, com foco em **Regressão Logística** e **Random Forest**.

O trabalho foi desenvolvido como **projeto acadêmico de Machine Learning**, seguindo um pipeline completo: EDA, pré-processamento, treinamento de modelos, ajuste de hiperparâmetros e avaliação.

---

## 📊 Problema e Dataset

O problema é formulado como uma **classificação binária**:

- `Churn = 1` → cliente cancelou o serviço  
- `Churn = 0` → cliente permaneceu ativo  

O dataset utilizado é o **Telco Customer Churn**, disponibilizado pela IBM:

- ~7.043 clientes  
- Variáveis de:
  - perfil do cliente (idade, dependentes, parceiro, etc.)
  - tempo de contrato (`tenure`)
  - tipo de serviços contratados (internet, telefone, TV, etc.)
  - tipo de contrato (mensal, 1 ano, 2 anos)
  - forma de pagamento
  - valores (`MonthlyCharges`, `TotalCharges`)
- Target: coluna `Churn` (Yes/No, mapeado para 1/0)

No código, o dataset é carregado diretamente pela URL pública da IBM (CSV).

---

## 🧠 Modelos Utilizados

Foram treinados e comparados dois modelos de ML clássico:

- **Modelo A – Regressão Logística**
  - `class_weight="balanced"`
  - Otimização do hiperparâmetro `C` com `GridSearchCV`
- **Modelo B – Random Forest**
  - `class_weight="balanced"`
  - Otimização de:
    - `n_estimators`
    - `max_depth`
    - `min_samples_split`

A métrica principal escolhida foi o **Recall da classe churn (1)**, pois errar um churn (falso negativo) é mais prejudicial para o negócio do que ter alguns falsos positivos.

---

## 🧬 Pipeline de Machine Learning

O pipeline completo implementado inclui:

1. **Carregamento e limpeza dos dados**
   - Conversão de `TotalCharges` para numérico
   - Remoção de linhas com `TotalCharges` nulo
   - Remoção de `customerID` (apenas identificador)

2. **Análise Exploratória (EDA)**
   - `info()`, `describe()`, contagem de nulos
   - Proporção de churn (dataset desbalanceado: ~73% não churn, ~27% churn)
   - Histogramas e gráficos simples (ex.: distribuição de `tenure`)

3. **Divisão treino/teste**
   - `train_test_split(test_size=0.3, stratify=y, random_state=42)`

4. **Pré-processamento com `ColumnTransformer`**
   - **Numéricas** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`):
     - `SimpleImputer(strategy="median")`
     - `StandardScaler()`
   - **Categóricas**:
     - `SimpleImputer(strategy="most_frequent")`
     - `OneHotEncoder(handle_unknown="ignore")`

5. **Treinamento com `Pipeline` + `GridSearchCV`**
   - Evita data leakage
   - Aplica pré-processamento + modelo em um único objeto

6. **Avaliação**
   - Métricas:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC AUC
   - Visualizações:
     - Matriz de confusão
     - Curva ROC
     - Curva Precision–Recall
   - Importância de atributos:
     - Coeficientes da Regressão Logística
     - `feature_importances_` da Random Forest

---

## 📈 Resultados (Resumo)

Desempenho no conjunto de teste:

| Modelo                | Accuracy | Precision | Recall | F1    | ROC AUC |
|-----------------------|----------|-----------|--------|-------|---------|
| Regressão Logística   | 0.736    | 0.503     | **0.793** | 0.615 | **0.838** |
| Random Forest         | 0.728    | 0.493     | 0.779 | 0.604 | 0.835   |

**Conclusão:**  
A **Regressão Logística** apresentou melhor desempenho geral no contexto do problema, com maior recall e maior área sob a curva ROC, além de ser mais interpretável.

Principais variáveis associadas ao churn:

- Tipo de contrato `Month-to-month`
- Baixo `tenure` (clientes novos)
- `InternetService = Fiber optic`
- `PaymentMethod = Electronic check`
- `TotalCharges` mais baixos (clientes com pouco tempo de casa)

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.x  
- **Bibliotecas principais:**
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/cristoffer21/churn-ML.git
cd churn-ML
