import torch
import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model).to(device)
        self.encoding.requires_grad = False  # 不需要训练

        pos = torch.arange(0, maxlen, device=device).float().unsqueeze(1)  # 位置索引 (maxlen, 1)
        _2i = torch.arange(0, d_model, 2, device=device).float()  # 偶数索引 (d_model // 2,)

        # 修复广播问题
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i.unsqueeze(0) / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i.unsqueeze(0) / d_model)))

    def forward(self, x):
        # x: 输入张量，形状为 (batch, seq_len, d_model)
        seq_len = x.shape[1]
        # 返回位置编码，扩展到 (batch, seq_len, d_model)
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(x.shape[0], 1, 1)

if __name__ == "__main__":
    batch = 2
    seq_len = 10
    d_model = 16
    vocab_size = 100
    maxlen = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TokenEmbedding 示例
    token_embedding = TokenEmbedding(vocab_size, d_model).to(device)
    tokens = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    token_emb = token_embedding(tokens)
    print("Token Embedding 输出形状:", token_emb.shape)  # (batch, seq_len, d_model)

    # PositionEmbedding 示例
    position_embedding = PositionEmbedding(d_model, maxlen, device)
    pos_emb = position_embedding(token_emb)
    print("Position Embedding 输出形状:", pos_emb.shape)  # (batch, seq_len, d_model)

    # 将 Token Embedding 和 Position Embedding 相加
    combined_emb = token_emb + pos_emb
    print("Combined Embedding 输出形状:", combined_emb.shape)  # (batch, seq_len, d_model)