import numpy as np
import random
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class TimeSeriesChunkGenerator:
    """
    현실적인 시계열 음악 추천 청크 데이터셋 생성기
    """
    
    def __init__(self, num_songs=10000, seed=42):
        self.num_songs = num_songs
        self.song_pool = [f"song_{i:05d}" for i in range(num_songs)]
        random.seed(seed)
        np.random.seed(seed)
        
        # 곡별 기본 인기도/특성 설정
        self.song_popularity = np.random.beta(2, 5, num_songs)  # 대부분 낮은 인기도
        self.song_genres = np.random.choice(['pop', 'rock', 'jazz', 'classical', 'electronic'], 
                                          num_songs, p=[0.3, 0.25, 0.15, 0.15, 0.15])
        
    def generate_target_patterns(self, num_targets=50):
        """
        타겟 곡들의 시계열 패턴 생성
        각 타겟 곡은 고유한 등장 패턴을 가짐
        """
        target_songs = random.sample(self.song_pool, num_targets)
        target_patterns = {}
        
        pattern_types = ['rising', 'falling', 'stable_high', 'volatile', 'periodic']
        
        for song in target_songs:
            pattern_type = random.choice(pattern_types)
            
            if pattern_type == 'rising':
                # 시간이 지날수록 상위권으로
                base_trend = np.linspace(0.7, 0.1, 100)  # 거리 감소 (상위권으로)
                
            elif pattern_type == 'falling':
                # 시간이 지날수록 하위권으로
                base_trend = np.linspace(0.1, 0.6, 100)
                
            elif pattern_type == 'stable_high':
                # 지속적으로 상위권
                base_trend = np.full(100, 0.15) + np.random.normal(0, 0.05, 100)
                
            elif pattern_type == 'volatile':
                # 변동성이 큰 패턴
                base_trend = 0.3 + 0.2 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
                
            else:  # periodic
                # 주기적 패턴
                base_trend = 0.25 + 0.15 * np.sin(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 0.03, 100)
            
            # 거리 값 정규화 (0.01~0.8 범위)
            base_trend = np.clip(base_trend, 0.01, 0.8)
            
            target_patterns[song] = {
                'type': pattern_type,
                'trend': base_trend,
                'appear_prob': max(0.3, self.song_popularity[self.song_pool.index(song)])
            }
        
        return target_patterns
    
    def generate_chunk(self, time_step: int, chunk_size: int, target_patterns: Dict, 
                      seasonal_factor: float = 1.0) -> List[Dict]:
        """
        특정 시점의 청크 생성
        """
        chunk = []
        
        # 타겟 곡들 중 이 시점에 등장할 곡들 선택
        appearing_targets = []
        for song, pattern in target_patterns.items():
            # 등장 확률 + 시간별 트렌드 고려
            appear_prob = pattern['appear_prob'] * seasonal_factor
            trend_index = min(time_step, len(pattern['trend']) - 1)
            
            if random.random() < appear_prob:
                target_distance = pattern['trend'][trend_index]
                appearing_targets.append({
                    'tid': song,
                    'dis': target_distance,
                    'is_target': True
                })
        
        # 타겟이 너무 많으면 일부만 선택
        if len(appearing_targets) > chunk_size // 2:
            appearing_targets = random.sample(appearing_targets, chunk_size // 2)
        
        chunk.extend(appearing_targets)
        
        # 나머지 슬롯을 일반 곡들로 채움
        remaining_slots = chunk_size - len(chunk)
        if remaining_slots > 0:
            # 타겟이 아닌 곡들 중에서 선택
            target_song_set = set(target_patterns.keys())
            non_target_songs = [s for s in self.song_pool if s not in target_song_set]
            
            selected_others = random.sample(non_target_songs, 
                                          min(remaining_slots, len(non_target_songs)))
            
            for song in selected_others:
                # 일반 곡들은 상대적으로 높은 거리값 (하위권)
                distance = np.random.beta(2, 3) * 0.7 + 0.2  # 0.2~0.9 범위
                chunk.append({
                    'tid': song,
                    'dis': distance,
                    'is_target': False
                })
        
        # 거리 기준으로 정렬
        chunk.sort(key=lambda x: x['dis'])
        
        # is_target 플래그 제거 (실제 사용시에는 없어야 함)
        final_chunk = [{'tid': item['tid'], 'dis': item['dis']} for item in chunk]
        
        return final_chunk, [item['is_target'] for item in chunk]
    
    def generate_dataset(self, num_chunks: int, chunk_size_range: Tuple[int, int] = (3, 8),
                        seasonal_cycles: int = 4, save_path: str = None) -> Tuple[List, Dict]:
        """
        대규모 시계열 청크 데이터셋 생성
        """
        print(f"=== {num_chunks}개 청크 데이터셋 생성 시작 ===")
        
        # 타겟 패턴 생성
        target_patterns = self.generate_target_patterns(num_targets=min(100, num_chunks // 20))
        print(f"타겟 곡 수: {len(target_patterns)}")
        
        chunks = []
        chunk_labels = []  # 각 청크의 타겟 곡 위치 정보
        
        # 계절성 효과
        seasonal_phase = np.linspace(0, seasonal_cycles * 2 * np.pi, num_chunks)
        
        for i in range(num_chunks):
            if (i + 1) % 1000 == 0:
                print(f"생성 진행률: {i+1}/{num_chunks} ({(i+1)/num_chunks*100:.1f}%)")
            
            # 청크 크기 랜덤 결정
            chunk_size = random.randint(*chunk_size_range)
            
            # 계절성 팩터 (1년 주기 등)
            seasonal_factor = 0.7 + 0.3 * (1 + np.sin(seasonal_phase[i])) / 2
            
            # 청크 생성
            chunk, target_flags = self.generate_chunk(i, chunk_size, target_patterns, seasonal_factor)
            
            chunks.append(chunk)
            chunk_labels.append(target_flags)
        
        print(f"✓ {len(chunks)}개 청크 생성 완료")
        
        # 데이터셋 통계
        total_songs = sum(len(chunk) for chunk in chunks)
        target_appearances = sum(sum(labels) for labels in chunk_labels)
        
        print(f"\n=== 데이터셋 통계 ===")
        print(f"총 청크 수: {len(chunks):,}")
        print(f"총 곡 등장 횟수: {total_songs:,}")
        print(f"타겟 곡 등장 횟수: {target_appearances:,}")
        print(f"평균 청크 크기: {total_songs/len(chunks):.2f}")
        print(f"타겟 등장 비율: {target_appearances/total_songs*100:.2f}%")
        
        # 파일 저장
        if save_path:
            dataset = {
                'chunks': chunks,
                'target_patterns': {k: {'type': v['type']} for k, v in target_patterns.items()},
                'metadata': {
                    'num_chunks': len(chunks),
                    'total_songs': total_songs,
                    'target_appearances': target_appearances,
                    'chunk_size_range': chunk_size_range
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"✓ 데이터셋 저장: {save_path}")
        
        return chunks, target_patterns, chunk_labels

# 사용 예제
def main():
    generator = TimeSeriesChunkGenerator(num_songs=5000)
    
    # 소규모 테스트 (100개)
    print("=== 소규모 테스트 ===")
    test_chunks, test_patterns, test_labels = generator.generate_dataset(
        num_chunks=100,
        chunk_size_range=(4, 6),
        save_path="small_dataset.json"
    )
    
    # 데이터 샘플 확인
    print(f"\n=== 데이터 샘플 ===")
    for i in range(min(5, len(test_chunks))):
        chunk = test_chunks[i]
        labels = test_labels[i]
        
        print(f"\n청크 {i+1}:")
        for j, (item, is_target) in enumerate(zip(chunk, labels)):
            target_mark = "★" if is_target else " "
            print(f"  {j+1}위: {item['tid']} (거리: {item['dis']:.3f}) {target_mark}")
    
    # 중규모 데이터셋 (1000개)
    print(f"\n=== 중규모 데이터셋 생성 ===")
    medium_chunks, medium_patterns, medium_labels = generator.generate_dataset(
        num_chunks=1000,
        chunk_size_range=(3, 8),
        save_path="medium_dataset.json"
    )
    
    # 대규모 데이터셋 (5000개) - 실제 사용 시
    print(f"\n=== 대규모 데이터셋 생성 ===")
    large_chunks, large_patterns, large_labels = generator.generate_dataset(
        num_chunks=5000,
        chunk_size_range=(3, 10),
        save_path="large_dataset.json"
    )
    
    return large_chunks, large_patterns, large_labels

# 메모리 효율적인 배치 로딩 함수
def load_chunks_in_batches(chunks: List, batch_size: int = 100):
    """
    대규모 데이터를 배치로 나누어 처리
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        yield batch, i // batch_size

# 실행
if __name__ == "__main__":
    chunks, patterns, labels = main()
    
    # 배치 처리 예제
    print(f"\n=== 배치 처리 예제 ===")
    batch_count = 0
    for batch_chunks, batch_idx in load_chunks_in_batches(chunks, batch_size=500):
        batch_count += 1
        print(f"배치 {batch_idx + 1}: {len(batch_chunks)}개 청크 처리")
        if batch_count >= 3:  # 처음 3개 배치만 출력
            break
    
    print(f"✓ 총 {len(chunks)}개 청크를 {batch_count}개 배치로 처리 가능")