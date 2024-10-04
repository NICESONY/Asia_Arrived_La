import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 적대적 예제 생성 함수 (여기서는 필요하지 않음)
def generate_image_adversary(model, image, label, eps=2 / 255.0):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int64)  # 레이블을 정수형으로 변환
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        loss = SparseCategoricalCrossentropy()(label, pred)
        gradient = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient)
        adversary = (image + (signed_grad * eps)).numpy()
        return adversary

# FGSM 이미지와 레이블 로드 함수
def load_images_and_labels(image_dir, label_dir, image_size=(64, 64)):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    images = []
    labels = []

    for image_file, label_file in zip(image_files, label_files):
        # 이미지 로드 및 전처리
        img_path = os.path.join(image_dir, image_file)
        image = load_img(img_path, target_size=image_size)
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        images.append(image)

        # 레이블 로드
        with open(os.path.join(label_dir, label_file), 'r') as file:
            label = int(file.read().strip())
            labels.append(label)

    return np.array(images), np.array(labels)

# 경로 설정
fgsm_image_path = './PARK/fgsmImage/images'
fgsm_label_path = './PARK/fgsmImage/labels'

# FGSM 이미지와 레이블 로드
fgsm_images, fgsm_labels = load_images_and_labels(fgsm_image_path, fgsm_label_path)

# 새로운 모델 정의
def create_model(input_shape=(64, 64, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# 새로운 모델 생성
model = create_model(input_shape=(64, 64, 3), num_classes=10)

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# FGSM 적대적 예제만 사용하여 새로운 모델 훈련
model.fit(fgsm_images, fgsm_labels, epochs=5, batch_size=32)

# 모델 평가
adversarial_loss, adversarial_accuracy = model.evaluate(fgsm_images, fgsm_labels)
print(f"적대적 예제에서의 손실: {adversarial_loss}")
print(f"적대적 예제에서의 정확도: {adversarial_accuracy}")

# 새로운 모델 저장
model.save('ShipClassifierNewAdversarialTrained.keras')
