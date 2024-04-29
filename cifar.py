import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # CelebA 데이터 로딩을 위해 TensorFlow를 사용합니다.

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


# 데이터셋 로딩 및 전처리
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()  # 임시로 CIFAR-10 데이터셋을 사용
x_train = x_train.astype('float32') / 255.
x_train = np.mean(x_train, axis=3, keepdims=True)  # 그레이스케일로 변환
x_train = x_train.reshape((x_train.shape[0], -1))  # 32x32x1 -> 1024

img_size = 32  # 이미지 크기 업데이트
n_in_out = img_size * img_size
n_mid = 256
n_z = 64

# 기존 클래스 및 함수는 재사용 가능
class BaseLayer:
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b


class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) * np.sqrt(2 / n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)  # ReLU

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)


class ParamsLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u  # 항등함수

    def backward(self, grad_y):
        delta = grad_y

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)


class OutputLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))  # sigmoid

    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)


class LatentLayer:
    def forward(self, mu, log_var):
        self.mu = mu
        self.log_var = log_var

        self.epsilon = np.random.randn(*log_var.shape)
        self.z = mu + self.epsilon * np.exp(log_var / 2)

    def backward(self, grad_z):
        self.grad_mu = grad_z + self.mu
        self.grad_log_var = grad_z * self.epsilon / 2 * np.exp(self.log_var / 2) - 0.5 * (1 - np.exp(self.log_var))

# 새로운 VAE 구조에 맞춘 전체 훈련 코드 추가
epochs = 201
batch_size = 32
eta = 0.001
interval = 20

# Encoder
middle_layer_enc = MiddleLayer(n_in_out, n_mid)
mu_layer = ParamsLayer(n_mid, n_z)
log_var_layer = ParamsLayer(n_mid, n_z)
z_layer = LatentLayer()

# Decoder
middle_layer_dec = MiddleLayer(n_z, n_mid)
output_layer = OutputLayer(n_mid, n_in_out)

def forward_propagation(x_mb):
    # Encoder
    middle_layer_enc.forward(x_mb)
    mu_layer.forward(middle_layer_enc.y)
    log_var_layer.forward(middle_layer_enc.y)
    z_layer.forward(mu_layer.y, log_var_layer.y)

    # Decoder
    middle_layer_dec.forward(z_layer.z)
    output_layer.forward(middle_layer_dec.y)


def backpropagation(t_mb):
    # Decoder
    output_layer.backward(t_mb)
    middle_layer_dec.backward(output_layer.grad_x)

    # Encoder
    z_layer.backward(middle_layer_dec.grad_x)
    log_var_layer.backward(z_layer.grad_log_var)
    mu_layer.backward(z_layer.grad_mu)
    middle_layer_enc.backward(mu_layer.grad_x + log_var_layer.grad_x)


def update_params():
    middle_layer_enc.update(eta)
    mu_layer.update(eta)
    log_var_layer.update(eta)
    middle_layer_dec.update(eta)
    output_layer.update(eta)

def get_rec_error(y, t):
    eps = 1e-7
    return -np.sum(t * np.log(y + eps) + (1 - t) * np.log(1 - y + eps)) / len(y)

def get_reg_error(mu, log_var):
    return -np.sum(1 + log_var - mu ** 2 - np.exp(log_var)) / len(mu)


n_batch = len(x_train) // batch_size

rec_error_record = []
reg_error_record = []
total_error_record = []

# 학습 루프
for i in range(epochs):
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    for j in range(n_batch):
        mb_index = index_random[j * batch_size:(j + 1) * batch_size]
        x_mb = x_train[mb_index, :]

        forward_propagation(x_mb)
        backpropagation(x_mb)

        update_params()

    if i % interval == 0 or i == epochs - 1:
        forward_propagation(x_train)

        rec_error = get_rec_error(output_layer.y, x_train)
        reg_error = get_reg_error(mu_layer.y, log_var_layer.y)
        total_error = rec_error + reg_error

        rec_error_record.append(rec_error)
        reg_error_record.append(reg_error)
        total_error_record.append(total_error)

        print(f"Epoch: {i} Rec_error: {rec_error}, Reg_error: {reg_error}, Total_error: {total_error}")

# 오차 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(rec_error_record) + 1), rec_error_record, label="Rec_Error")
plt.plot(range(1, len(reg_error_record) + 1), reg_error_record, label="Reg_Error")
plt.plot(range(1, len(total_error_record) + 1), total_error_record, label="Total_Error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()