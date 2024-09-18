import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import os
import random
import tensorflow as tf

# 클래스 이름
ClassNames = ['Aircraft Carrier', 'Bulkers', 'Car Carrier', 'Container Ship', 'Cruise', 'DDG', 'Recreational', 'Sailboat', 'Submarine', 'Tug']

# 데이터 경로 설정
def load_data(image_path, label_path):
    data = []
    for file in os.listdir(image_path):
        file_p = os.path.join(image_path, file)
        image = cv2.imread(file_p)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        data.append(image / 255.0)  # 정규화
    labels = []
    for file in os.listdir(label_path):
        file_p = os.path.join(label_path, file)
        with open(file_p, "r") as f:
            labels.append(int(f.read(1)))
    return np.array(data), np.array(labels)

# 훈련 데이터 로드
base_path = 'C:/dev/workspace/sec-ai/Asia_Arrived_La/Data/Ships_dataset/train'
data_image = os.path.join(base_path, 'images')
data_label = os.path.join(base_path, 'labels')
images, labels = load_data(data_image, data_label)

# 검증 데이터 로드
base_path_valid = 'C:/dev/workspace/sec-ai/Asia_Arrived_La/Data/Ships_dataset/valid'
data_image_valid = os.path.join(base_path_valid, 'images')
data_label_valid = os.path.join(base_path_valid, 'labels')
validationimages, validationlabels = load_data(data_image_valid, data_label_valid)

# 모델 정의
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(images, labels, batch_size=32, epochs=20, validation_data=(validationimages, validationlabels))

# 모델 저장
model.save('ShipClassifierV1.keras') 

# 모델 로드
model = models.load_model('ShipClassifierV1.keras')

# FGSM 공격 생성 함수
def fgsm_attack(image, label, model, epsilon):
    image = tf.convert_to_tensor(image[None, ...])  # 배치 차원 추가
    label = tf.convert_to_tensor(label[None])  # 배치 차원 추가
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversarial_image = image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    return adversarial_image[0].numpy()  # 배치 차원 제거 및 numpy 배열로 변환

# BIM 공격 생성 함수
def bim_attack(image, label, model, epsilon, alpha, iterations):
    adversarial_image = image
    for i in range(iterations):
        adversarial_image = fgsm_attack(adversarial_image, label, model, alpha)
        perturbation = np.clip(adversarial_image - image, -epsilon, epsilon)
        adversarial_image = np.clip(image + perturbation, 0, 1)
    return adversarial_image

# 테스트 데이터 로드
base_path_test = 'C:/dev/workspace/sec-ai/Asia_Arrived_La/Data/Ships_dataset/test'
data_image_test = os.path.join(base_path_test, 'images')
data_label_test = os.path.join(base_path_test, 'labels')
testimages, testlabels = load_data(data_image_test, data_label_test)

# BIM 공격 적용
epsilon = 0.1  # 공격 강도
alpha = 0.01  # BIM의 단계 크기
iterations = 10  # BIM 반복 횟수
bim_adversarial_images = np.array([bim_attack(image, label, model, epsilon, alpha, iterations) for image, label in zip(testimages, testlabels)])

# 원본 및 BIM 공격된 이미지 비교
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(bim_adversarial_images[i], cmap=plt.cm.binary)
    pred = model.predict(np.array([bim_adversarial_images[i]]))  # 배치 차원 추가
    index = np.argmax(pred)
    plt.xlabel(f"Actual = {ClassNames[testlabels[i]]} \n Predicted = {ClassNames[index]}")
plt.show()

# 모델 평가
loss, accuracy = model.evaluate(testimages, testlabels)
print(f"Loss on test data: {loss}")
print(f"Accuracy on test data: {accuracy}")

loss, accuracy_bim_adversarial = model.evaluate(bim_adversarial_images, testlabels)
print(f"Loss on BIM adversarial data: {loss}")
print(f"Accuracy on BIM adversarial data: {accuracy_bim_adversarial}")

# Adversarial Training을 위한 데이터 생성
adv_images = np.concatenate((images, bim_adversarial_images))
adv_labels = np.concatenate((labels, testlabels))

# 모델 다시 훈련 (Adversarial Training)
model.fit(adv_images, adv_labels, batch_size=32, epochs=10, validation_data=(validationimages, validationlabels))

# Adversarial Training 후 모델 평가
loss, accuracy = model.evaluate(testimages, testlabels)
print(f"Loss on test data after adversarial training: {loss}")
print(f"Accuracy on test data after adversarial training: {accuracy}")

loss, accuracy_bim_adversarial = model.evaluate(bim_adversarial_images, testlabels)
print(f"Loss on BIM adversarial data after adversarial training: {loss}")
print(f"Accuracy on BIM adversarial data after adversarial training: {accuracy_bim_adversarial}")

#결과---> 확실히 BIM 과 FGSM 학습을 통해 적대적 예제 학습이 됨
