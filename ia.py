import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mplcyberpunk
import yfinance as yf

plt.style.use("cyberpunk")

# modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=150, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=150, hidden_size=150, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=150, hidden_size=150, batch_first=True)
        self.fc = nn.Linear(150, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.fc(x)
        return x

# menu principal
while True:
    print("\n===== MENU =====")
    print("[1] Prever cotação de uma empresa")
    print("[2] Sair")
    escolha = input("Escolha uma opção: ")

    if escolha == "1":
        empresa = input("Digite o código da empresa (Ex: MGLU3.SA, PETR4.SA, VALE3.SA, ...): ").upper()
        try:
            ticker = yf.Ticker(empresa)
            dados = ticker.history(period="5y")

            if dados.empty:
                print("❌ Código não encontrado ou sem dados disponíveis. Verifique se foi digitado corretamente.")
                continue

            dados = dados[["Close"]].rename(columns={"Close": "fechamento"})
            dados["fechamento_dia_seguinte"] = dados["fechamento"].shift(-1)
            dados = dados.dropna()
            dados.index = pd.to_datetime(dados.index)

            tamanho_dados_treinamento = int(len(dados) * 0.8)

            escalador_treinamento = MinMaxScaler(feature_range=(0, 1))
            escalador_teste = MinMaxScaler(feature_range=(0, 1))

            dados_entre_0_e_1_treinamento = escalador_treinamento.fit_transform(dados.iloc[0:tamanho_dados_treinamento])
            dados_entre_0_e_1_teste = escalador_teste.fit_transform(dados.iloc[tamanho_dados_treinamento:])

            x_treinamento = dados_entre_0_e_1_treinamento[:, 0].reshape(-1, 1, 1)
            y_treinamento = dados_entre_0_e_1_treinamento[:, 1].reshape(-1, 1, 1)

            x_teste = dados_entre_0_e_1_teste[:, 0].reshape(-1, 1, 1)
            y_teste = dados_entre_0_e_1_teste[:, 1].reshape(-1, 1, 1)

            x_treinamento_tensor = torch.tensor(x_treinamento, dtype=torch.float32)
            y_treinamento_tensor = torch.tensor(y_treinamento, dtype=torch.float32)
            x_teste_tensor = torch.tensor(x_teste, dtype=torch.float32)

            train_dataset = TensorDataset(x_treinamento_tensor, y_treinamento_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            print("\nTreinando modelo...")

            for epoch in range(10):
                running_loss = 0.0
                for x_batch, y_batch in train_loader:
                    output = model(x_batch)
                    loss = criterion(output, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Época {epoch + 1}, Loss médio: {running_loss / len(train_loader):.4f}")

            with torch.no_grad():
                precos_preditos_tensor = model(x_teste_tensor)

            precos_preditos = precos_preditos_tensor.numpy().reshape(-1, 1)

            dados_teste = np.concatenate((x_teste.reshape(-1, 1), y_teste.reshape(-1, 1)), axis=1)
            dados_preditos = np.concatenate((x_teste.reshape(-1, 1), precos_preditos), axis=1)

            precos_teste_reais = escalador_teste.inverse_transform(dados_teste)
            precos_teste_preditos = escalador_teste.inverse_transform(dados_preditos)

            datas_teste = dados.index[tamanho_dados_treinamento:]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(datas_teste, precos_teste_reais[:, 1], label="Real")
            ax.plot(datas_teste, precos_teste_preditos[:, 1], label="Previsão da IA")

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            plt.title(f"Histórico cotação {empresa} X Previsão da IA")
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("❌ Ocorreu um erro ao buscar os dados ou durante a execução.")
            print("Detalhes:", str(e))

    elif escolha == "2":
        print("Obrigado por usar nosso sistema. Até logo!")
        break

    else:
        print("⚠️ Por favor, selecione uma opção válida (1 ou 2).")
