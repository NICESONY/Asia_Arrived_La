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
model.save('ShipClassifierV1.keras')  # 권장되는 포맷으로 저장

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

# 테스트 데이터 로드
base_path_test = 'C:/dev/workspace/sec-ai/Asia_Arrived_La/Data/Ships_dataset/test'
data_image_test = os.path.join(base_path_test, 'images')
data_label_test = os.path.join(base_path_test, 'labels')
testimages, testlabels = load_data(data_image_test, data_label_test)

# FGSM 공격 적용
epsilon = 0.1  # 공격 강도
adversarial_images = np.array([fgsm_attack(image, label, model, epsilon) for image, label in zip(testimages, testlabels)])

# 원본 및 공격된 이미지 비교
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(adversarial_images[i], cmap=plt.cm.binary)
    pred = model.predict(np.array([adversarial_images[i]]))  # 배치 차원 추가
    index = np.argmax(pred)
    plt.xlabel(f"Actual = {ClassNames[testlabels[i]]} \n Predicted = {ClassNames[index]}")
plt.show()

# 모델 평가
loss, accuracy = model.evaluate(testimages, testlabels)
print(f"Loss on test data: {loss}")
print(f"Accuracy on test data: {accuracy}")

loss, accuracy_adversarial = model.evaluate(adversarial_images, testlabels)
print(f"Loss on adversarial data: {loss}")
print(f"Accuracy on adversarial data: {accuracy_adversarial}")
