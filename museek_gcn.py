import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 시계열 청크 분석 기반 음악 추천 GCN
class TimeSeriesRecommendationGCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_songs):
        super(TimeSeriesRecommendationGCN, self).__init__()
        self.num_songs = num_songs
        
        # 각 청크 내 곡들의 관계를 학습하는 GCN
        self.chunk_gc1 = nn.Linear(in_features, hidden_features)
        self.chunk_gc2 = nn.Linear(hidden_features, hidden_features)
        
        # 청크 간의 시계열 관계를 학습하는 레이어
        self.temporal_lstm = nn.LSTM(hidden_features, hidden_features, batch_first=True)
        
        # 최종 곡 추천을 위한 분류기
        self.final_classifier = nn.Linear(hidden_features, num_songs)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, chunk_features, chunk_adjs, chunk_sizes):
        # chunk_features: (num_chunks, max_chunk_size, in_features)
        # chunk_adjs: (num_chunks, max_chunk_size, max_chunk_size)
        # chunk_sizes: 각 청크의 실제 곡 수
        
        num_chunks = chunk_features.shape[0]
        chunk_representations = []
        
        # 각 청크별로 GCN 적용하여 청크 표현 생성
        for i in range(num_chunks):
            # 현재 청크의 특징과 인접 행렬
            x = chunk_features[i].unsqueeze(0)  # (1, max_chunk_size, in_features)
            adj = chunk_adjs[i].unsqueeze(0)    # (1, max_chunk_size, max_chunk_size)
            
            # GCN 레이어 1
            x = torch.matmul(adj, x)
            x = self.chunk_gc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # GCN 레이어 2  
            x = torch.matmul(adj, x)
            x = self.chunk_gc2(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # 청크 표현: 실제 곡들만 고려한 평균 풀링
            chunk_size = chunk_sizes[i]
            chunk_repr = torch.mean(x[0, :chunk_size, :], dim=0)  # (hidden_features,)
            chunk_representations.append(chunk_repr)
        
        # 청크 표현들을 시계열로 stack
        chunk_sequence = torch.stack(chunk_representations).unsqueeze(0)  # (1, num_chunks, hidden_features)
        
        # LSTM으로 시계열 패턴 학습
        lstm_out, (hidden, _) = self.temporal_lstm(chunk_sequence)
        
        # 마지막 히든 스테이트를 사용하여 최종 곡 예측
        final_representation = hidden[-1].squeeze(0)  # (hidden_features,)
        
        # 최종 곡 추천 점수
        song_scores = self.final_classifier(final_representation)  # (num_songs,)
        
        return song_scores