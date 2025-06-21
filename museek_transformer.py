import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRecommendationTransformer(nn.Module):
    def __init__(self, song_vocab_size, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        
        # 임베딩 레이어들
        self.song_embedding = nn.Embedding(song_vocab_size + 1, embed_dim)  # +1 for "no song" token
        self.position_embedding = nn.Embedding(100, embed_dim)  # 최대 100개 청크
        self.dis_projection = nn.Linear(1, embed_dim)  # dis 값을 임베딩 차원으로
        
        # "추천하지 않음" 토큰을 위한 특별 임베딩
        self.no_recommendation_token = nn.Parameter(torch.randn(1, embed_dim))
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 레이어 (이진 분류 + 곡 선택)
        self.should_recommend = nn.Linear(embed_dim, 1)  # 추천할지 말지
        self.song_selection = nn.Linear(embed_dim, 1)    # 어떤 곡을 선택할지
        
    def forward(self, song_ids, dis_values, chunk_positions, attention_mask):
        """
        Args:
            song_ids: [batch_size, max_songs] - 모든 후보곡 ID들 (0은 "no recommendation" 토큰)
            dis_values: [batch_size, max_songs, 1] - 각 곡의 dis 값
            chunk_positions: [batch_size, max_songs] - 각 곡이 나온 청크 위치
            attention_mask: [batch_size, max_songs] - 패딩 마스크
        """
        batch_size, max_songs = song_ids.shape
        
        # "추천하지 않음" 토큰 추가
        no_rec_token = self.no_recommendation_token.expand(batch_size, 1, -1)
        no_rec_pos = torch.zeros(batch_size, 1, dtype=torch.long, device=song_ids.device)
        no_rec_dis = torch.zeros(batch_size, 1, 1, device=dis_values.device)
        no_rec_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=attention_mask.device)
        
        # 기존 임베딩 계산
        song_embeds = self.song_embedding(song_ids)  # [batch_size, max_songs, embed_dim]
        pos_embeds = self.position_embedding(chunk_positions)  # [batch_size, max_songs, embed_dim]
        dis_embeds = self.dis_projection(dis_values)  # [batch_size, max_songs, embed_dim]
        
        # 모든 임베딩 결합
        combined_embeds = song_embeds + pos_embeds + dis_embeds
        
        # "추천하지 않음" 토큰과 결합
        all_embeds = torch.cat([no_rec_token, combined_embeds], dim=1)
        all_masks = torch.cat([no_rec_mask, attention_mask.bool()], dim=1)
        
        # Transformer 통과
        transformer_output = self.transformer(
            all_embeds, 
            src_key_padding_mask=~all_masks
        )
        
        # 이진 분류: 추천할지 말지
        should_recommend_logits = self.should_recommend(transformer_output[:, 0])  # [batch_size, 1]
        
        # 곡 선택 점수
        song_scores = self.song_selection(transformer_output[:, 1:])  # [batch_size, max_songs, 1]
        song_scores = song_scores.squeeze(-1)  # [batch_size, max_songs]
        
        # 마스킹된 위치는 매우 낮은 점수로
        song_scores = song_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        return {
            'should_recommend': should_recommend_logits.squeeze(-1),  # [batch_size]
            'song_scores': song_scores  # [batch_size, max_songs]
        }

# 데이터 전처리 함수 - 10x10 형태 처리
def prepare_data_10x10(chunk_recommendations_batch):
    """
    Args:
        chunk_recommendations_batch: List of 10x10 형태 데이터
        각 샘플: 10개 청크 × 각 청크당 최대 10개 후보곡
        예시: [
            [[(song_id, dis), (song_id, dis), ...], [...], ...],  # 첫 번째 샘플 (10개 청크)
            [[(song_id, dis), (song_id, dis), ...], [...], ...],  # 두 번째 샘플
            ...
        ]
    
    Returns:
        딕셔너리 형태의 배치 데이터
    """
    batch_size = len(chunk_recommendations_batch)
    num_chunks = 10
    max_songs_per_chunk = 10
    max_total_songs = num_chunks * max_songs_per_chunk
    
    batch_song_ids = []
    batch_dis_values = []
    batch_chunk_positions = []
    batch_attention_masks = []
    
    for sample_idx, sample in enumerate(chunk_recommendations_batch):
        # 각 샘플의 모든 곡 정보 수집
        sample_songs = []
        sample_dis = []
        sample_positions = []
        
        for chunk_idx in range(num_chunks):
            if chunk_idx < len(sample):
                chunk_songs = sample[chunk_idx]
                # 각 청크에서 최대 max_songs_per_chunk만큼만 처리
                for song_idx, (song_id, dis_value) in enumerate(chunk_songs[:max_songs_per_chunk]):
                    sample_songs.append(song_id)
                    sample_dis.append([dis_value])
                    sample_positions.append(chunk_idx)
        
        # 패딩 처리 (max_total_songs까지)
        current_length = len(sample_songs)
        if current_length < max_total_songs:
            sample_songs.extend([0] * (max_total_songs - current_length))
            sample_dis.extend([[0.0]] * (max_total_songs - current_length))
            sample_positions.extend([0] * (max_total_songs - current_length))
        else:
            # 길이가 초과하면 자르기
            sample_songs = sample_songs[:max_total_songs]
            sample_dis = sample_dis[:max_total_songs]
            sample_positions = sample_positions[:max_total_songs]
            current_length = max_total_songs
        
        # attention mask 생성
        attention_mask = [1] * current_length + [0] * (max_total_songs - current_length)
        
        batch_song_ids.append(sample_songs)
        batch_dis_values.append(sample_dis)
        batch_chunk_positions.append(sample_positions)
        batch_attention_masks.append(attention_mask)
    
    return {
        'song_ids': torch.tensor(batch_song_ids),
        'dis_values': torch.tensor(batch_dis_values, dtype=torch.float),
        'chunk_positions': torch.tensor(batch_chunk_positions),
        'attention_mask': torch.tensor(batch_attention_masks)
    }

# 학습 코드 - 10x10 형태 데이터용
def train_model():
    # 모델 초기화
    model = MusicRecommendationTransformer(song_vocab_size=10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 두 개의 손실 함수
    bce_loss = nn.BCEWithLogitsLoss()  # 추천할지 말지
    ce_loss = nn.CrossEntropyLoss()    # 어떤 곡을 선택할지
    
    # 10x10 형태의 가상 학습 데이터
    train_data = [
        # 첫 번째 샘플: 10개 청크, 각 청크마다 후보곡들
        [
            [(1, 0.1), (2, 0.3), (3, 0.5)],           # 청크 0
            [(2, 0.2), (4, 0.4)],                     # 청크 1  
            [(1, 0.15), (5, 0.6)],                    # 청크 2
            [(6, 0.3), (7, 0.4), (8, 0.5)],          # 청크 3
            [(9, 0.2), (10, 0.35)],                   # 청크 4
            [(11, 0.4), (12, 0.5)],                   # 청크 5
            [(13, 0.1), (14, 0.3)],                   # 청크 6
            [(15, 0.25), (16, 0.4)],                  # 청크 7
            [(17, 0.3), (18, 0.45)],                  # 청크 8
            [(19, 0.2), (20, 0.35)]                   # 청크 9
        ],
        # 두 번째 샘플: 좋은 추천곡이 있는 경우
        [
            [(21, 0.05), (22, 0.1)],                  # 매우 좋은 dis 값
            [(23, 0.08), (24, 0.12)],
            [(25, 0.06), (26, 0.11)],
            [(27, 0.09), (28, 0.13)],
            [(29, 0.07), (30, 0.14)],
            [(31, 0.10), (32, 0.15)],
            [(33, 0.08), (34, 0.12)],
            [(35, 0.09), (36, 0.13)],
            [(37, 0.11), (38, 0.16)],
            [(39, 0.10), (40, 0.14)]
        ],
        # 세 번째 샘플: 추천할 만한 곡이 없는 경우 (모든 dis 값이 높음)
        [
            [(41, 0.8), (42, 0.9)],                   # 나쁜 dis 값들
            [(43, 0.85), (44, 0.92)],
            [(45, 0.88), (46, 0.95)],
            [(47, 0.82), (48, 0.89)],
            [(49, 0.87), (50, 0.94)],
            [(51, 0.83), (52, 0.91)],
            [(53, 0.86), (54, 0.93)],
            [(55, 0.84), (56, 0.90)],
            [(57, 0.89), (58, 0.96)],
            [(59, 0.85), (60, 0.92)]
        ]
    ]
    
    # 정답 레이블 (직접 학습시킴)
    labels = [
        {'should_recommend': 1, 'song_idx': 0},  # 첫 번째 샘플: 추천함, 첫 번째 곡 선택
        {'should_recommend': 1, 'song_idx': 2},  # 두 번째 샘플: 추천함, 세 번째 곡 선택  
        {'should_recommend': 0, 'song_idx': -1}, # 세 번째 샘플: 추천하지 않음
    ]
    
    model.train()
    for epoch in range(100):
        # 배치 데이터 준비 (10x10 형태 처리)
        batch_data = prepare_data_10x10(train_data)
        
        # 레이블 준비
        should_recommend_labels = torch.tensor([label['should_recommend'] for label in labels], dtype=torch.float)
        song_labels = []
        for i, label in enumerate(labels):
            if label['should_recommend'] == 1:
                song_labels.append(label['song_idx'])
            else:
                song_labels.append(0)  # 추천하지 않을 때는 dummy 값
        song_labels = torch.tensor(song_labels)
        
        # Forward pass
        outputs = model(
            batch_data['song_ids'],
            batch_data['dis_values'], 
            batch_data['chunk_positions'],
            batch_data['attention_mask']
        )
        
        # Loss 계산
        # 1. 추천할지 말지에 대한 손실
        recommend_loss = bce_loss(outputs['should_recommend'], should_recommend_labels)
        
        # 2. 곡 선택에 대한 손실 (추천하는 경우만)
        song_loss = 0
        recommend_mask = should_recommend_labels == 1
        if recommend_mask.sum() > 0:
            song_loss = ce_loss(
                outputs['song_scores'][recommend_mask], 
                song_labels[recommend_mask]
            )
        
        total_loss = recommend_loss + song_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            song_loss_val = song_loss.item() if isinstance(song_loss, torch.Tensor) else song_loss
            total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            recommend_loss_val = recommend_loss.item()
            print(f'Epoch {epoch}, Total Loss: {total_loss_val:.4f}, Recommend Loss: {recommend_loss_val:.4f}, Song Loss: {song_loss_val:.4f}')

    return model

# 추론 함수 - 10x10 형태 데이터용 
def predict_10x10(model, chunk_recommendations_10x10, threshold=0.5):
    """
    10x10 형태 데이터로 추천곡 선택
    
    Args:
        chunk_recommendations_10x10: 10개 청크 × 각 청크당 최대 10개 후보곡
        예시: [[(song_id, dis), ...], [(song_id, dis), ...], ...]
    """
    model.eval()
    with torch.no_grad():
        batch_data = prepare_data_10x10([chunk_recommendations_10x10])
        outputs = model(
            batch_data['song_ids'],
            batch_data['dis_values'],
            batch_data['chunk_positions'], 
            batch_data['attention_mask']
        )
        
        # 1. 먼저 추천할지 말지 결정
        should_recommend_prob = torch.sigmoid(outputs['should_recommend'][0]).item()
        
        if should_recommend_prob < threshold:
            return {
                'song_id': None,
                'confidence': should_recommend_prob,
                'recommendation': '추천하지 않음'
            }
        
        # 2. 추천한다면 어떤 곡을 선택할지
        best_song_idx = torch.argmax(outputs['song_scores'][0]).item()
        
        # 실제 곡 ID와 해당 청크 찾기
        all_songs = []
        song_to_chunk = {}
        
        for chunk_idx, chunk_songs in enumerate(chunk_recommendations_10x10):
            for song_id, dis_value in chunk_songs:
                all_songs.append(song_id)
                song_to_chunk[len(all_songs) - 1] = {
                    'chunk_idx': chunk_idx,
                    'song_id': song_id,
                    'dis_value': dis_value
                }
        
        if best_song_idx < len(all_songs):
            selected_info = song_to_chunk[best_song_idx]
            return {
                'song_id': selected_info['song_id'],
                'chunk_idx': selected_info['chunk_idx'],
                'dis_value': selected_info['dis_value'],
                'confidence': should_recommend_prob
            }
        else:
            return {
                'song_id': None,
                'confidence': should_recommend_prob,
                'recommendation': '인덱스 오류'
            }

# 사용 예시
if __name__ == "__main__":
    # 학습
    model = train_model()
    
    # 예측 예시
    test_chunk_recommendations = [
        [(1234, 0.1), (1237, 0.3)],  # 청크 0의 후보곡들
        [(1235, 0.2), (1238, 0.4)],  # 청크 1의 후보곡들
        [(1236, 0.15), (1, 0.5)]  # 청크 2의 후보곡들
    ]
    recommended_song = predict_10x10(model, test_chunk_recommendations)
    print(f"추천된 곡 ID: {recommended_song}")