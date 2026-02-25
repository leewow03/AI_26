import ssl
import pandas as pd
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
# 로지스틱 회귀- 소프트멕스
# 롤체 전적을 활용해 시즌 티어 예측 하기

# 특성
# ['avg_rank', 'top4_rate', 'games', 'avg_max_level']
# 평균 등수 (실력의 결과)
# 순방 확률 (안정성)
# 판수 (데이터 신뢰도)
# 평균 최대 레벨 (운영 능력)

lolche = pd.read_csv('https://bit.ly/4s7QYRZ', sep='\s+')

lolche_input = lolche[['avg_rank', 'top4_rate', 'games', 'avg_max_level']].to_numpy()
lolche_target = lolche[['avg_rank', 'top4_rate', 'games', 'avg_max_level']].to_numpy()


train_input, test_input, train_target, test_target = train_test_split(lolche_input, lolche_target)

print(train_input.shape, test_input.shape)


















