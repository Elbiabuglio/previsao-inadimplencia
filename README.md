# Previsao Inadimplencia




## Estrutura do projeto
- `src/` - código Python
- `notebooks/` - exploração e análises
- `config/` - parâmetros do pipeline
- `.env` - variáveis de ambiente
- `tests/` - testes unitários

## Como rodar
1. Criar e ativar virtualenv
2. Instalar dependências: `pip install -r requirements.txt`
3. Configurar `.env` com suas credenciais
4. Rodar scripts Python:
```bash
python src/extract.py
python src/transform.py
python src/load.py