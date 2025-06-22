# 📈 Previsor de Cotações com LSTM (PyTorch)

<p align="center">
  <img src="https://github.com/Amanda-FL-CAOS/ia-bolsa-de-valores/blob/main/exemplo_gr%C3%A1fico.png" alt="exemplo de gráfico" width="700">
</p>

Este é um projeto em Python que utiliza redes neurais LSTM para prever a cotação do dia seguinte de ações da bolsa de valores brasileira (B3), com base em dados históricos fornecidos pela API do Yahoo Finance.

---

## 🧠 Sobre o Projeto

Este sistema permite ao usuário digitar o código de uma empresa listada na B3 (como `PETR4.SA`, `VALE3.SA`, etc.) e, a partir dos dados dos últimos 5 anos, treina uma rede neural LSTM para prever a cotação de fechamento do dia seguinte.

O projeto utiliza:
- `yfinance` para obtenção de dados financeiros
- `PyTorch` para criação e treinamento do modelo
- `MinMaxScaler` para normalização dos dados
- `matplotlib` com estilo *cyberpunk* para visualização dos resultados

---

## 🛠 Tecnologias Usadas

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- mplcyberpunk
- Scikit-learn
- PyTorch
- yfinance

---

## 🧪 Como Usar

### 1. Clone o repositório

```bash
git clone https://github.com/seuusuario/nome-do-repositorio.git
cd nome-do-repositorio
2. Instale as dependências
bash
Copiar
Editar
pip install -r requirements.txt
Ou instale manualmente:

bash
Copiar
Editar
pip install pandas numpy matplotlib mplcyberpunk scikit-learn torch yfinance
3. Execute o programa
bash
Copiar
Editar
python nome_do_arquivo.py
🖼 Exemplo de Uso
text
Copiar
Editar
===== MENU =====
[1] Prever cotação de uma empresa
[2] Sair
Escolha uma opção: 1
Digite o código da empresa (Ex: MGLU3.SA, PETR4.SA, VALE3.SA, ...): PETR4.SA
Após o treinamento, será exibido um gráfico parecido com o que está no topo do arquivo.

📊 Modelo LSTM
A rede neural utilizada contém:

3 camadas LSTM empilhadas (hidden_size=150)

1 camada totalmente conectada (Linear)

O modelo é treinado por 10 épocas usando:

Função de perda: MSELoss

Otimizador: Adam

⚠️ Aviso
Este projeto tem fins educacionais e experimentais. Não é recomendado para uso em operações reais de investimento, pois não considera variáveis externas fundamentais, notícias ou contexto de mercado.

📃 Licença
Este projeto está licenciado sob a MIT License.

🙋‍♀️ Contribuindo
Contribuições são bem-vindas! Sinta-se livre para abrir uma issue ou enviar um pull request com melhorias ou sugestões.

✨ Autor(a)
Desenvolvido com 💻 e 📈 por Amanda Fernandes.
