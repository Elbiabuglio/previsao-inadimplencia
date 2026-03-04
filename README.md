📊 Previsão de Inadimplência

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-pipeline%20local-green.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint](https://img.shields.io/badge/lint-flake8-blue.svg)](https://flake8.pycqa.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()

Projeto para construção de um **modelo preditivo de inadimplência**, desde a análise exploratória até a execução automatizada do pipeline em produção.

---

## 🎯 Objetivo

Desenvolver um modelo de Machine Learning capaz de prever a probabilidade de um cliente se tornar inadimplente, utilizando:

- Engenharia de atributos
- One-Hot Encoding
- Balanceamento da variável alvo (SMOTE)
- Regressão Logística
- Avaliação com ROC AUC e métricas de classificação

---

## 🏗 Arquitetura do Projeto
```mermaid
flowchart LR
    A[Banco SQL Server] --> B[database.py]
    B --> C[preprocessing.py]
    C --> D[modeling.py]
    D --> E[Modelo Treinado .pkl]
```

### Fluxo do Pipeline

1. 🔄 **Carregamento dos dados** do banco
2. 🧹 **Pré-processamento**
3. 🏗 **Engenharia de Features**
4. 🔢 **One-Hot Encoding**
5. ⚖ **Balanceamento da Target** (SMOTE)
6. 🤖 **Treinamento do Modelo**
7. 📈 **Avaliação**
8. 💾 **Salvamento do Modelo**

---

## 📁 Estrutura do Projeto
```
previsao-inadimplencia/
│
├── src/
│   ├── database.py
│   ├── preprocessing.py
│   └── modeling.py
│
├── notebooks/
│   └── analysis_data.ipynb
│
├── models/
│   └── modelo_inadimplencia.pkl
│
├── tests/
├── config/
├── main.py
├── requirements.txt
└── .env
```

---

## ⚙️ Tecnologias Utilizadas

- **Python** 3.13
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Machine Learning
- **Imbalanced-learn** - SMOTE para balanceamento
- **Joblib** - Serialização de modelos
- **SQL Server** - Banco de dados (Docker)
- **SQLAlchemy** - ORM e conexão com banco
- **Jupyter Notebook** - Análise exploratória

---

## 🐳 Banco de Dados (Docker)

O projeto utiliza SQL Server rodando em container Docker.

### docker-compose.yml
```yaml
version: "3.9"

services:
  sqlserver:
    image: mcr.microsoft.com/mssql/server:2022-latest
    container_name: sqlserver
    environment:
      SA_PASSWORD: ${SA_PASSWORD}
      ACCEPT_EULA: "Y"
    ports:
      - "1433:1433"
    volumes:
      - sqlserver-data:/var/opt/mssql
    restart: unless-stopped

volumes:
  sqlserver-data:
```

---

## 🔄 Integração com Banco de Dados no Pipeline

O pipeline de scoring em produção inclui as seguintes etapas automatizadas de manipulação de dados no banco SQL Server:

### Etapas do Processo

1. **Conexão com o Banco**: A conexão é feita via SQLAlchemy utilizando variáveis de ambiente configuradas no arquivo `.env`. Isso garante segurança e flexibilidade para diferentes ambientes.

2. **Limpeza da Tabela Intermediária**: Antes de inserir as novas previsões, a tabela `RESULTADOS_INTERMEDIARIOS` é limpa com o comando SQL `TRUNCATE TABLE`, evitando dados duplicados.

3. **Inserção das Previsões**: Os dados gerados pelo modelo (número do contrato, previsão e probabilidade de inadimplência) são inseridos na tabela `RESULTADOS_INTERMEDIARIOS` usando `pandas.DataFrame.to_sql` com controle de `chunksize` para otimizar a inserção.

4. **Execução da Stored Procedure**: Após a inserção, a procedure `dbo.SP_INPUT_RESULTADOS_MODELO_PREDITIVO` é executada para processar os resultados no banco, podendo realizar atualizações, cálculos ou integrações adicionais.

### Como Isso Está Implementado

No arquivo `src/database.py` temos funções específicas:

- `truncate_table(table_name)` — limpa a tabela com `TRUNCATE TABLE`
- `insert_dataframe(df, table_name, if_exists='append')` — insere os dados no banco com controle de tamanho de lotes
- `execute_procedure(procedure_name)` — executa a stored procedure no banco
- `get_engine()` — cria a engine de conexão segura com o banco
- `test_connection()` — valida se a conexão está funcionando

### Exemplo de Uso no Pipeline

No `main.py`, no pipeline de scoring, essas funções são chamadas assim:
```python
# Limpa tabela intermediária
insert_dataframe(df_resultado, "RESULTADOS_INTERMEDIARIOS", if_exists="replace")

# Executa procedure final
execute_procedure("dbo.SP_INPUT_RESULTADOS_MODELO_PREDITIVO")
```

Isso garante que os dados estejam sempre atualizados e processados para uso em produção.

---

## 🔐 Variáveis de Ambiente (.env)

Crie um arquivo `.env` na raiz do projeto:
```env
DB_HOST=localhost
DB_PORT=1433
DB_USER=usuario_python
DB_PASS=sua_senha_segura
DB_NAME=modelos_preditivos

SA_PASSWORD=sua_senha_sa
```

> ⚠️ **Importante:** O arquivo `.env` não deve ser versionado no Git.

---

## 🚀 Como Executar o Projeto

### 1️⃣ Criar ambiente virtual
```bash
python -m venv .venv
```

**Ativar o ambiente:**

- **Linux / Mac:**
```bash
  source .venv/bin/activate
```

- **Windows:**
```bash
  .venv\Scripts\activate
```

### 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Executar pipeline completo
```bash
python main.py
```

**O sistema irá:**

1. Conectar ao banco
2. Pré-processar dados
3. Aplicar One-Hot Encoding
4. Balancear com SMOTE
5. Treinar modelo de Regressão Logística
6. Avaliar desempenho (ROC AUC)
7. Salvar modelo em `/models`
8. Inserir resultados no banco
9. Executar stored procedure de processamento

---

## 📊 Métricas do Modelo

Exemplo de resultado obtido:

| Métrica | Valor |
|---------|-------|
| **ROC AUC** | ≈ 0.81 |
| **F1-Score** (classe inadimplente) | ≈ 0.90 |
| **Modelo** | Regressão Logística |
| **Balanceamento** | SMOTE (apenas treino) |

---

## 📓 Notebooks

### `analysis_data.ipynb`

Contém:

- ✅ Análise exploratória
- ✅ Tratamento de valores missing
- ✅ Criação de faixas de financiamento
- ✅ Testes de balanceamento
- ✅ Visualizações

---

## 🧠 Pipeline de Machine Learning

### Pré-processamento (`preprocessing.py`)

- Criação de variáveis derivadas
- Tratamento de datas
- Tratamento de valores ausentes
- One-Hot Encoding
- Separação de X e y

### Modelagem (`modeling.py`)

- Train/Test Split
- Aplicação de SMOTE
- Padronização com StandardScaler
- Regressão Logística
- Avaliação com ROC AUC
- Salvamento do modelo

### Banco de Dados (`database.py`)

- Conexão segura via SQLAlchemy
- Funções de limpeza de tabelas
- Inserção otimizada de DataFrames
- Execução de stored procedures
- Validação de conexão

---

## 🧪 Próximas Melhorias

- [ ] Implementar Cross-Validation
- [ ] Testar Random Forest / XGBoost
- [ ] Criar API com FastAPI
- [ ] Implementar monitoramento de modelo
- [ ] Deploy em nuvem
- [ ] Adicionar logs estruturados
- [ ] Implementar alertas de performance

---

## 👩‍💻 Autor

Projeto desenvolvido para prática completa de Engenharia de Machine Learning, do banco de dados até produção.

---

## 📌 Versão

**v1.0.0** – Pipeline completo com SMOTE integrado e integração automatizada com banco de dados

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
