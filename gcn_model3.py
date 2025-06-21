import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PositionalGCN(nn.Module):
    """
    곡 ID에 의존하지 않고 위치적/구조적 특성만으로 학습하는 GCN
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(PositionalGCN, self).__init__()
        self.gc1 = nn.Linear(in_features, hidden_features)
        self.gc2 = nn.Linear(hidden_features, hidden_features)
        self.gc3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, adj):
        # x: [batch_size, num_nodes, features]
        # adj: [batch_size, num_nodes, num_nodes]
        
        # 첫 번째 GCN 레이어
        h1 = torch.relu(self.gc1(torch.bmm(adj, x)))
        h1 = self.dropout(h1)
        
        # 두 번째 GCN 레이어
        h2 = torch.relu(self.gc2(torch.bmm(adj, h1)))
        h2 = self.dropout(h2)
        
        # 출력 레이어
        out = self.gc3(torch.bmm(adj, h2))
        
        return out

class TimeSeriesPositionalRecommendation(nn.Module):
    """
    시계열 위치 패턴 기반 추천 모델
    """
    def __init__(self, node_features=3, hidden_features=64, time_features=32):
        super(TimeSeriesPositionalRecommendation, self).__init__()
        
        # 각 청크의 그래프 패턴 학습
        self.gcn = PositionalGCN(node_features, hidden_features, hidden_features)
        
        # 시계열 패턴 학습
        self.lstm = nn.LSTM(hidden_features, time_features, batch_first=True, bidirectional=True)
        
        # 시계열 표현을 위한 레이어
        self.time_encoder = nn.Sequential(
            nn.Linear(time_features * 2, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 노드별 추천 점수 예측
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),  # 시계열 + 노드 특성
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, chunk_features, chunk_adjs, chunk_sizes, predict_chunk_idx=None):
        batch_size = chunk_features.size(0)
        
        # 각 청크별 GCN 처리
        chunk_embeddings = []
        all_node_embeddings = []
        
        for i in range(batch_size):
            size = chunk_sizes[i]
            features = chunk_features[i:i+1, :size, :]
            adj = chunk_adjs[i:i+1, :size, :size]
            
            # GCN으로 노드 임베딩 생성
            node_embeddings = self.gcn(features, adj)  # [1, size, hidden]
            all_node_embeddings.append(node_embeddings)
            
            # 청크 전체 표현
            chunk_repr = torch.mean(node_embeddings, dim=1)  # [1, hidden]
            chunk_embeddings.append(chunk_repr)
        
        # 시계열 임베딩 생성
        time_embeddings = torch.cat(chunk_embeddings, dim=0).unsqueeze(0)  # [1, num_chunks, hidden]
        
        # LSTM으로 시계열 패턴 학습
        lstm_out, _ = self.lstm(time_embeddings)  # [1, num_chunks, time_features*2]
        
        # 전체 시계열 표현
        time_repr = self.time_encoder(torch.mean(lstm_out, dim=1))  # [1, hidden]
        
        # 특정 청크의 노드들에 대한 추천 점수 계산
        if predict_chunk_idx is not None:
            target_nodes = all_node_embeddings[predict_chunk_idx].squeeze(0)  # [size, hidden]
            num_nodes = target_nodes.size(0)
            
            # 시계열 표현을 각 노드와 결합
            time_repr_expanded = time_repr.expand(num_nodes, -1)  # [size, hidden]
            combined = torch.cat([target_nodes, time_repr_expanded], dim=1)  # [size, hidden*2]
            
            # 각 노드의 추천 점수
            node_scores = self.node_scorer(combined).squeeze(1)  # [size]
            return node_scores
        else:
            # 훈련 시에는 전체 패턴 점수만 반환
            return torch.mean(time_repr)

def preprocess_positional_chunks(chunks):
    """
    위치 기반 특성만 추출 (곡 ID 제거)
    """
    max_chunk_size = max(len(chunk) for chunk in chunks)
    num_chunks = len(chunks)
    
    # 특징: [거리, 순위_정규화, 거리_정규화]만 사용 (곡 ID 제외)
    chunk_features = np.zeros((num_chunks, max_chunk_size, 3))
    chunk_adjs = np.zeros((num_chunks, max_chunk_size, max_chunk_size))
    chunk_sizes = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        chunk_sizes.append(chunk_size)
        
        # 해당 청크에서의 거리 범위
        distances = [item['dis'] for item in chunk]
        min_dist, max_dist = min(distances), max(distances)
        dist_range = max_dist - min_dist if max_dist > min_dist else 1
        
        for i, song_data in enumerate(chunk):
            # 특징 1: 원본 거리
            chunk_features[chunk_idx, i, 0] = song_data['dis']
            
            # 특징 2: 청크 내 순위 (0~1 정규화, 0이 최상위)
            chunk_features[chunk_idx, i, 1] = i / max(chunk_size - 1, 1)
            
            # 특징 3: 청크 내 거리 정규화 (0~1)
            normalized_dist = (song_data['dis'] - min_dist) / dist_range
            chunk_features[chunk_idx, i, 2] = normalized_dist
        print(chunks)
        print(chunk_features)
        
        # 인접 행렬: 거리와 순위 유사도 기반
        for i in range(chunk_size):
            for j in range(chunk_size):
                if i == j:
                    chunk_adjs[chunk_idx, i, j] = 1.0
                else:
                    # 거리 유사도
                    dist_sim = np.exp(-abs(chunk[i]['dis'] - chunk[j]['dis']) * 5)
                    # 순위 유사도
                    rank_sim = np.exp(-abs(i - j) * 0.5)
                    # 결합 유사도
                    chunk_adjs[chunk_idx, i, j] = (dist_sim + rank_sim) / 2
    
    return chunk_features, chunk_adjs, chunk_sizes

# 실험 데이터: 곡 ID는 단순 placeholder
chunks = [
    # 시점 1: 타겟이 최상위 (ID는 의미없음)
    [{'tid': 'song_X', 'dis': 0.1}, {'tid': 'song_Y', 'dis': 0.2}, {'tid': 'song_Z', 'dis': 0.25}],
    
    # 시점 2: 타겟이 여전히 상위권
    [{'tid': 'song_A', 'dis': 0.12}, {'tid': 'song_X', 'dis': 0.15}, {'tid': 'song_B', 'dis': 0.22}],
    
    # 시점 3: 타겟이 중위권으로
    [{'tid': 'song_C', 'dis': 0.11}, {'tid': 'song_X', 'dis': 0.28}, {'tid': 'song_D', 'dis': 0.35}],
    
    # 시점 4: 타겟이 다시 최상위
    [{'tid': 'song_X', 'dis': 0.08}, {'tid': 'song_E', 'dis': 0.19}, {'tid': 'song_F', 'dis': 0.31}],
    
    # 시점 5: 타겟 없음 (다른 패턴)
    [{'tid': 'song_G', 'dis': 0.14}, {'tid': 'song_H', 'dis': 0.21}, {'tid': 'song_I', 'dis': 0.33}]
]

# 학습 타겟: 패턴 분석 결과 상위권 등장 가능성이 높음
target_pattern = 1.0  # 상위권 등장 패턴

print("=== 위치 기반 시계열 청크 데이터 ===")
for i, chunk in enumerate(chunks):
    songs_info = [f"pos{j+1}({item['dis']:.2f})" for j, item in enumerate(chunk)]
    has_target = any('song_X' in item['tid'] for item in chunk)
    target_pos = next((j+1 for j, item in enumerate(chunk) if 'song_X' in item['tid']), None)
    status = f"← X는 {target_pos}위" if has_target else ""
    print(f"시점 {i+1}: {songs_info} {status}")

print(f"\n학습 목표: 시계열 패턴으로 상위권 등장 가능성 예측")

# 전처리
chunk_features, chunk_adjs, chunk_sizes = preprocess_positional_chunks(chunks)

print("===========")
print(chunk_adjs)

exit(1)

# 텐서 변환
features_tensor = torch.FloatTensor(chunk_features)
adjs_tensor = torch.FloatTensor(chunk_adjs)
target_tensor = torch.FloatTensor([target_pattern])

# 모델 생성
model = TimeSeriesPositionalRecommendation(
    node_features=3,
    hidden_features=32,
    time_features=16
)

# 학습용 타겟 생성 (각 청크에서 타겟 곡의 위치)
def create_training_targets(chunks, target_song='song_X'):
    """각 청크에서 타겟 곡의 위치 정보를 생성"""
    targets = []
    for chunk in chunks:
        target_positions = []
        for i, item in enumerate(chunk):
            if target_song in item['tid']:
                # 타겟 곡이면 높은 점수 (상위권일수록 더 높음)
                position_score = 1.0 - (i / len(chunk))  # 0위=1.0, 1위=0.67, 2위=0.33
                target_positions.append(position_score)
            else:
                target_positions.append(0.1)  # 비타겟 곡은 낮은 점수
        targets.append(target_positions)
    return targets

training_targets = create_training_targets(chunks, 'song_X')

# 학습
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
print(f"\n=== 위치 패턴 학습 ===")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # 각 청크별로 노드 점수 예측
    for chunk_idx, target_scores in enumerate(training_targets):
        if len(target_scores) > 0:  # 빈 청크가 아닐 때만
            node_scores = model(features_tensor, adjs_tensor, chunk_sizes, predict_chunk_idx=chunk_idx)
            target_tensor = torch.FloatTensor(target_scores[:len(node_scores)])
            
            loss = criterion(node_scores, target_tensor)
            total_loss += loss
    
    avg_loss = total_loss / len(training_targets)
    
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss.item():.6f}')

print(f"\n=== 새로운 데이터 테스트 ===")
# 완전히 새로운 곡들로만 구성된 테스트 데이터
test_chunks = [
    [{'tid': 'new_song_1', 'dis': 0.09}, {'tid': 'new_song_2', 'dis': 0.18}, {'tid': 'new_song_3', 'dis': 0.27}],
    [{'tid': 'new_song_1', 'dis': 0.11}, {'tid': 'new_song_5', 'dis': 0.16}, {'tid': 'new_song_6', 'dis': 0.24}],
    [{'tid': 'new_song_1', 'dis': 0.13}, {'tid': 'new_song_8', 'dis': 0.29}, {'tid': 'new_song_9', 'dis': 0.35}],
]

print("테스트 청크:")
for i, chunk in enumerate(test_chunks):
    songs_info = [f"{item['tid']}({item['dis']:.2f})" for item in chunk]
    print(f"시점 {i+1}: {songs_info}")

test_features, test_adjs, test_sizes = preprocess_positional_chunks(test_chunks)
test_features_tensor = torch.FloatTensor(test_features)
test_adjs_tensor = torch.FloatTensor(test_adjs)

model.eval()
with torch.no_grad():
    print(f"\n=== 각 청크별 추천 결과 ===")
    
    for chunk_idx, chunk in enumerate(test_chunks):
        # 해당 청크의 각 곡에 대한 추천 점수
        node_scores = model(test_features_tensor, test_adjs_tensor, test_sizes, predict_chunk_idx=chunk_idx)
        
        # 점수와 곡 정보 결합
        song_scores = []
        for i, (song_data, score) in enumerate(zip(chunk, node_scores)):
            song_scores.append({
                'song': song_data['tid'],
                'distance': song_data['dis'],
                'position': i + 1,
                'recommendation_score': score.item()
            })
        
        # 추천 점수로 정렬
        song_scores.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        print(f"\n청크 {chunk_idx + 1} 추천 결과:")
        for i, song_info in enumerate(song_scores):
            print(f"  {i+1}위: {song_info['song']} "
                  f"(원래 {song_info['position']}위, 거리: {song_info['distance']:.2f}, "
                  f"추천점수: {song_info['recommendation_score']:.4f})")
        
        best_recommendation = song_scores[0]
        print(f"  → 최종 추천: {best_recommendation['song']} "
              f"(추천점수: {best_recommendation['recommendation_score']:.4f})")

    # 전체 테스트 데이터에서 최고 추천곡 찾기
    all_recommendations = []
    for chunk_idx, chunk in enumerate(test_chunks):
        node_scores = model(test_features_tensor, test_adjs_tensor, test_sizes, predict_chunk_idx=chunk_idx)
        for i, (song_data, score) in enumerate(zip(chunk, node_scores)):
            all_recommendations.append({
                'song': song_data['tid'],
                'chunk': chunk_idx + 1,
                'score': score.item()
            })
    
    # 전체에서 최고 점수
    all_recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n=== 전체 테스트 데이터 최종 추천 ===")
    print(f"🎵 추천 곡: {all_recommendations[0]['song']} "
          f"(청크 {all_recommendations[0]['chunk']}, 점수: {all_recommendations[0]['score']:.4f})")
    
    print(f"\n상위 3개 추천:")
    for i, rec in enumerate(all_recommendations[:3]):
        print(f"  {i+1}위: {rec['song']} (청크 {rec['chunk']}, 점수: {rec['score']:.4f})")

print(f"\n=== 모델 특징 ===")
print("✓ 곡 ID에 의존하지 않음 - 완전히 새로운 곡도 처리 가능")
print("✓ 거리, 순위, 위치적 특성만으로 학습")
print("✓ 그래프 구조와 시계열 패턴을 동시에 고려")
print("✓ 실제 서비스에서 모르는 곡들에도 적용 가능")