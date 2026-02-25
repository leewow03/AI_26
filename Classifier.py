import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # 분류

# 분류 가까운 이웃을 보고 다수결로 1인지, 2인지 판단함(다수결임으로 짝수이면 안됨)
# 회귀 가까운 이웃을 합쳐 평균을 내 예측되는 수치를 뽑아냄

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 1: 도미, 0: 빙어

# 총 데이터 값
fish_data = np.column_stack((fish_length, fish_weight)) #자동으로 2차원

# 정답 알려주기
fish_target = np.concatenate((np.ones(35),np.zeros(14)))

# 80% 훈련용으로 설정후 특성 섞기, 그후 랜덤하게 섞기
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, train_size= 0.8, stratify=fish_target, random_state=42)

#훈련용(39객체, 2특성) | 테스트용(10객체, 2특성)
# print(train_input.shape, test_input.shape)

# 훈련용 세트 표준점수 구하기
mean=np.mean(train_input, axis=0)   # 평균을 구한다. axis : 중심선
std=np.std(train_input, axis=0)

# 표준 점수로 전처리 수행
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std  # 테스트 세트도 훈련 세트 기준으로!

# 모델 학습 (전처리된 데이터로 다시)
kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)

# 4. 모델 평가 (괄호 사용 주의!)
accuracy = kn.score(test_scaled, test_target)
print(f"정확도: {accuracy}")
# 현재는 데이터가 별로 없어서 1.0이 나옴
# 정확도는  % 가 아닌 비율임

# 5. 새로운 데이터 예측
new = ([25, 150] - mean) / std
prediction = kn.predict([new])

if prediction == 1:
    print("예측 결과: 도미입니다!")
else:
    print("예측 결과: 빙어입니다!")

# 1. 'new' 물고기의 이웃 5마리를 찾습니다.
distances, indexes = kn.kneighbors([new])

plt.title('preproced data graph') # 그래프 제목
# 2. 산점도를 그립니다.
plt.scatter(train_scaled[:,0], train_scaled[:,1], label='train') # 전체 훈련 데이터
plt.scatter(new[0], new[1], marker='^', s=100, label='new')      # 예측하려는 데이터 (세모)

# 3. 이웃 데이터만 뽑아서 다른 색(초록색)으로 덧칠합니다.
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D', color='green', label='neighbors')

plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()




















