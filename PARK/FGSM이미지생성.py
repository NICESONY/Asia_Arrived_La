import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import save_img

# 적대적 예제 생성 함수
def generate_image_adversary(model, image, label, eps=0.1):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int64)
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image, training=False)
        loss = SparseCategoricalCrossentropy()(label, pred)
        gradient = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient)
        adversary = image + (signed_grad * eps)
        adversary = tf.clip_by_value(adversary, 0, 1)  # 이미지 값이 0~1 범위 내에 있도록 클리핑
        return adversary.numpy()

# 저장 경로 설정
base_path = "./PARK/fgsmImage"  
image_save_path = os.path.join(base_path, 'images')
label_save_path = os.path.join(base_path, 'labels')
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)

# 모델 로드
model_path = 'ShipClassifierV1.h5'  
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")

model = tf.keras.models.load_model(model_path)

# 이미지와 라벨 데이터 준비 (여기서는 `images`와 `labels`가 이미 로드되어 있다고 가정합니다.)
# 예시 데이터 로드 (실제 데이터 로드 코드로 교체 필요)
images = np.random.rand(2000, 64, 64, 3)  # 실제 이미지 데이터로 교체
labels = np.random.randint(0, 10, size=(2000,))  # 실제 라벨 데이터로 교체

adversarial_examples = []
selected_labels = []

# 테스트 이미지에 대해 적대적 예제 생성 및 저장
for i in range(2000):  # 테스트 이미지가 2000장 있다고 가정
    image = images[i]
    label = labels[i]
    adversary = generate_image_adversary(model, image.reshape(1, 64, 64, 3), np.array([label]), eps=0.1)
    
    adversarial_examples.append(adversary[0])  # [0] 인덱스를 사용하여 (64, 64, 3) 형상으로 변환
    selected_labels.append(label)

    # 이미지 저장
    image_file_path = os.path.join(image_save_path, f'adversary_{i}.png')
    save_img(image_file_path, adversary[0])
    
    # 라벨 저장
    label_file_path = os.path.join(label_save_path, f'label_{i}.txt')
    with open(label_file_path, 'w') as label_file:
        label_file.write(str(label))

# 적대적 예제를 리스트에서 numpy 배열로 변환
adversarial_examples = np.array(adversarial_examples)
selected_labels = np.array(selected_labels)

# 적대적 예제를 사용하여 모델의 손실과 정확도 평가
adversarial_loss, adversarial_accuracy = model.evaluate(adversarial_examples, selected_labels)
print(f"적대적 테스트 데이터에서의 손실: {adversarial_loss}")
print(f"적대적 테스트 데이터에서의 정확도: {adversarial_accuracy}")
