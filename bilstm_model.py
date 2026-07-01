import torch
import torch.nn as nn
import config

class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, embedding_dim=config.EMBEDDING_DIM, hidden_dim=256, output_dim=5):
        super(EmotionBiLSTM, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        
        # 3. Fully Connected Classification Head (hidden_dim * 2 because it's bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text):
        # text shape: [batch_size, seq_length]
        embedded = self.dropout(self.embedding(text))
        
        # lstm_out shape: [batch_size, seq_length, hidden_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Extract the final hidden state from both forward and backward directions
        # Concat the final forward hidden layer (hidden[-2,:,:]) and backward hidden layer (hidden[-1,:,:])
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through linear layer to get raw logits
        logits = self.fc(self.dropout(hidden_last))
        return logits

if __name__ == "__main__":
    # Test with a dummy batch to ensure shapes match up correctly
    model = EmotionBiLSTM()
    dummy_input = torch.randint(0, config.VOCAB_SIZE, (config.BATCH_SIZE, config.MAX_SEQUENCE_LENGTH))
    output = model(dummy_input)
    print("🚀 BiLSTM Model verified successfully!")
    print(f"Input batch shape : {dummy_input.shape}")
    print(f"Output logits shape: {output.shape} (Batch Size, Number of Emotions)")