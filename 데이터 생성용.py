import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. 출력 제한 해제 (300행을 모두 보여주기 위해)
pd.set_option('display.max_rows', None)

# 2. 유형별 데이터 생성 (각 100명씩)
np.random.seed(42)

# 유형 1: 벽에 막힌 자 (정체형) - Target 0
type1 = np.random.normal([4.6, 0.48, 350, 7.9], 0.1, (100, 4))
y1 = [0] * 100

# 유형 2: 고속 승급형 (재능충) - Target 2
type2 = np.random.normal([2.8, 0.78, 25, 8.8], 0.1, (100, 4))
y2 = [2] * 100

# 유형 3: 점진적 승급형 (노력파) - Target 1
type3 = np.random.normal([4.0, 0.58, 110, 8.2], 0.1, (100, 4))
y3 = [1] * 100

# 3. 데이터 합치기
X_total = np.vstack([type1, type2, type3])
y_total = np.array(y1 + y2 + y3)

columns = ['avg_rank', 'top4_rate', 'games', 'avg_max_level']
lolche_df = pd.DataFrame(X_total, columns=columns)
lolche_df['target'] = y_total

# 4. 전체 데이터 출력
print("--- 300명의 롤체 학습 데이터 전체 목록 ---")
print(lolche_df)

# 5. 로지스틱 회귀 모델 학습 (이후 예측을 위해)
model = LogisticRegression(max_iter=1000)
model.fit(X_total, y_total)