import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 적대적 예제 생성 함수 (BIM)
def generate_image_bim_adversary(model, image, label, eps=2 / 255.0, iterations=10, alpha=0.01):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int64)  # 레이블을 정수형으로 변환

    adv_image = image

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            pred = model(adv_image)
            loss = SparseCategoricalCrossentropy()(label, pred)
            gradient = tape.gradient(loss, adv_image)
            signed_grad = tf.sign(gradient)
            adv_image = adv_image + alpha * signed_grad
            adv_image = tf.clip_by_value(adv_image, image - eps, image + eps)  # 변형된 이미지가 원래 이미지와 너무 멀어지지 않도록 제한
            adv_image = tf.clip_by_value(adv_image, 0, 1)  # 이미지 값이 [0, 1] 범위를 벗어나지 않도록 제한
    return adv_image.numpy()

# BIM 이미지와 레이블 로드 함수
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
bim_image_path = './PARK/bimImage/images'
bim_label_path = './PARK/bimImage/labels'

# BIM 이미지와 레이블 로드
bim_images, bim_labels = load_images_and_labels(bim_image_path, bim_label_path)

# 모델 로드
model_path = 'ShipClassifierV1.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")

model = tf.keras.models.load_model(model_path)

# BIM 적대적 예제 생성
adversarial_examples = []
adversarial_labels = []

for image, label in zip(bim_images, bim_labels):
    adversary = generate_image_bim_adversary(model, image.reshape(1, 64, 64, 3), np.array([label]), eps=0.1, iterations=10, alpha=0.01)
    adversarial_examples.append(adversary[0])
    adversarial_labels.append(label)

# 적대적 예제와 레이블 배열로 변환
adversarial_examples = np.array(adversarial_examples)
adversarial_labels = np.array(adversarial_labels)

# 원본 훈련 데이터 로드 (여기서는 예시로 mnist 데이터셋을 사용)
# 실제 데이터로 교체 필요
train_images = np.random.rand(2000, 64, 64, 3)  # 실제 이미지 데이터로 교체
train_labels = np.random.randint(0, 10, size=(2000,))  # 실제 레이블 데이터로 교체

# 적대적 예제만 사용하여 새로운 모델 훈련
combined_images = adversarial_examples
combined_labels = adversarial_labels

# 모델 재훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(combined_images, combined_labels, epochs=5, batch_size=32)

# 모델 평가
# 적대적 예제에 대한 평가
adversarial_loss, adversarial_accuracy = model.evaluate(adversarial_examples, adversarial_labels)
print(f"적대적 예제에서의 손실: {adversarial_loss}")
print(f"적대적 예제에서의 정확도: {adversarial_accuracy}")

# 모델 저장
model.save('ShipClassifierBIMTrained.h5')
