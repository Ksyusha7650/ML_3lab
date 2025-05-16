import json
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка и подготовка данных
with open('data/poems.json', 'r', encoding='utf-8') as f:
    poems = json.load(f)

# Собираем все стихи в один текст
text = '\n'.join(['\n'.join(lines) for lines in poems.values()]).lower()

# Создаем словарь символов
tokens = sorted(set(text))
num_tokens = len(tokens)
idx_to_token = {i: ch for i, ch in enumerate(tokens)}
token_to_idx = {ch: i for i, ch in enumerate(tokens)}

# Кодируем текст
encoded = torch.tensor([token_to_idx[ch] for ch in text], dtype=torch.long)

# Создаем Dataset
class PoemDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return x, y

seq_length = 100
batch_size = 32
dataset = PoemDataset(encoded, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Модель LSTM
class PoemLSTM(nn.Module):
    def __init__(self, num_tokens, embedding_dim=128, hidden_dim=384, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tokens)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden):
        batch_size = x.size(0)  # Получаем реальный размер батча
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(-1, out.size(2)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

model = PoemLSTM(num_tokens).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение
def train(model, dataloader, epochs=3):
    print(f'train')
    model.train()
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0))  # Инициализируем hidden для текущего батча
            
            hidden = tuple(h.detach() for h in hidden)
            optimizer.zero_grad()
            
            output, hidden = model(x, hidden)
            loss = criterion(output, y.reshape(-1))
            
            loss.backward()
            optimizer.step()
        total_loss = 0
        for x, y in dataloader:
            ...
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    
        #print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Генерация текста
def generate(model, start_str, length=200, temperature=0.5):
    model.eval()
    chars = [ch for ch in start_str.lower()]
    hidden = model.init_hidden(1)
    
    for ch in start_str:
        x = torch.tensor([[token_to_idx[ch]]]).to(device)
        _, hidden = model(x, hidden)
    
    for _ in range(length):
        x = torch.tensor([[token_to_idx[chars[-1]]]]).to(device)
        output, hidden = model(x, hidden)
        
        probs = F.softmax(output / temperature, dim=-1).squeeze()
        char_idx = torch.multinomial(probs, 1).item()
        chars.append(idx_to_token[char_idx])
    
    return ''.join(chars)

# Пример генерации
train(model, dataloader)
print(generate(model, "люблю тебя", length=300))