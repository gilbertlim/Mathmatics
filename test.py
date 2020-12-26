
import numpy as np
import matplotlib.pyplot as plt

# -- 입력과 정답 준비--
input_data = np.arange(0, np.pi * 2, 0.1)  # 입력
correct_data = np.sin(input_data)  # 정답
input_data = (input_data - np.pi) / np.pi  # 입력을 -1.0 ~ 1.0 범위 안으로 설정
n_data = len(correct_data)  # 데이터 수

# --각 설정 값--
n_in = 1  # 입력층의 뉴런 수
n_mid = 3  # 은닉층의 뉴런 수
n_out = 1  # 출력층의 뉴런 수

wb_width = 0.01  # 가중치와 편향 설정을 위한 정규분포의 표준 편차
eta = 0.1  # 학습률
epoch = 2001
interval = 200  # 경과 표시 간격

# --은닉층--
class MiddleLayer:  # 초기 설정
    def __init__(self, n_upper, n):
        # n_upper : 앞 층의 뉴런 수
        # n : 현재 층의 뉴런 수

        # 가중치(n_upper x n 행렬)
        self.w = wb_width * np.random.randn(n_upper, n)

        # 편향(n 벡터)
        self.b = wb_width * np.random.randn(n)

    def forward(self, x):  # 순전파
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))  # 시그모이드 함수

    def backward(self, grad_y):  # 역전파
        delta = grad_y * (1 - self.y) * self.y  # 시그모이드 함수 미분

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, eta):  # 가중치와 편향 수정
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# --출력층--
class OutputLayer:
    def __init__(self, n_upper, n):  # 초기 설정
        # n_upper : 앞 층의 뉴런 수
        # n : 현재 층의 뉴런 수

        # 가중치(n_upper x n 행렬)
        self.w = wb_width * np.random.randn(n_upper, n)
        # wb_width : 정규분포의 표준편차
        # np.random.rand() : 정규분포를 따르는 난수 생성, float

        # 편향(n 벡터)
        self.b = wb_width * np.random.randn(n)

    def forward(self, x):  # 순전파
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u  # 항등 함수

    def backward(self, t):  # 역전파
        delta = self.y - t  # 오차제곱합
        # t : 정답

        # 계산된 기울기들
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)

    def update(self, eta):  # 가중치와 편향 수정
        # 확률적 경사하강법
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# --각 층의 초기화(앞 층의 뉴런 수, 해당 층의 뉴런 수)--
middle_layer = MiddleLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

# --학습(x 에포크 수)--
for i in range(epoch):
    # 인덱스 셔플(섞기)
    index_random = np.arange(n_data)  # 0 ~ (n_data - 1)
    np.random.shuffle(index_random)

    # 결과 표시
    total_error = 0
    plot_x = []
    plot_y = []

    for idx in index_random:
        x = input_data[idx: idx + 1]  # 입력
        t = correct_data[idx: idx + 1]  # 정답

        # 순전파
        # 입력을 행렬로 변환(온라인 학습, 배치사이즈 1)
        middle_layer.forward(x.reshape(1, 1))
        output_layer.forward(middle_layer.y)

        # 역전파
        # 정답을 행렬로 변환(온라인 학습, 배치사이즈 1)
        output_layer.backward(t.reshape(1, 1))
        middle_layer.backward(output_layer.grad_x)

        # 가중치와 편향 수정
        middle_layer.update(eta)
        output_layer.update(eta)

        if i % interval == 0:
            y = output_layer.y.reshape(-1)  # 행렬을 벡터로 되돌림

            # 오차제곱합 계산
            total_error += 1.0 / 2.0 * np.sum(np.square(y - t))

            # 출력 기록
            plot_x.append(x)
            plot_y.append(y)

    if i % interval == 0:
        # 출력 그래프 표시
        plt.plot(input_data, correct_data, linestyle="dashed")
        plt.scatter(plot_x, plot_y, marker="+")
        plt.show()

        # 에포크 수와 오차 표시
        print("Epoch:" + str(i) + "/" + str(epoch), "Error:" + str(total_error / n_data))