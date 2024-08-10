import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pyimagesearch.fgsm import generate_image_adversary


# 데이터셋 경로 설정
train_dir = os.path.join('PARK', 'dataset', 'Ships dataset', 'train', 'images')

val_dir = os.path.join('PARK', 'dataset', 'Ships dataset', 'valid', 'images')


# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10개 클래스
])

# 모델 컴파일
print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 모델 훈련
print("[INFO] training network...")
model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)

# 적대적 훈련 함수 정의
def adversarial_training(model, train_generator, val_generator, eps=0.01, epochs=10):
    for epoch in range(epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{epochs}...")
        for i in range(len(train_generator)):
            # 배치 데이터 가져오기
            x_batch, y_batch = train_generator.next()

            # 적대적 예제 생성
            x_adversarial = generate_image_adversary(model, x_batch, y_batch, eps=eps)

            # 원본 이미지와 적대적 이미지를 결합
            combined_images = np.vstack([x_batch, x_adversarial])
            combined_labels = np.vstack([y_batch, y_batch])

            # 결합된 데이터로 모델 학습
            model.train_on_batch(combined_images, combined_labels)

        # 에폭이 끝날 때마다 검증 데이터로 평가
        val_loss, val_acc = model.evaluate(val_generator, verbose=0)
        print(f"[INFO] Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

# 적대적 훈련 수행
adversarial_training(model, train_generator, val_generator, eps=0.01, epochs=10)

# 적대적 샘플 생성 및 시각화
for i in np.random.choice(np.arange(0, len(val_generator)), size=(10,)):
    # 현재 이미지와 레이블 가져오기
    x_batch, y_batch = val_generator.next()
    image = x_batch[i]
    label = y_batch[i]

    # 적대적 샘플 생성 및 예측
    adversary = generate_image_adversary(model, image.reshape(1, 224, 224, 3), label, eps=0.1)
    pred = model.predict(adversary)

    # 적대적 샘플을 시각화
    adversary = adversary.reshape((224, 224, 3)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image * 255
    image = image.astype("uint8")

    # 이미지를 크기 조정하여 더 잘 시각화
    image = cv2.resize(image, (224, 224))
    adversary = cv2.resize(adversary, (224, 224))

    # 원본 이미지와 적대적 이미지에 대한 예측 레이블 결정
    imagePred = label.argmax()
    adversaryPred = pred[0].argmax()

    # 이미지와 적대적 이미지에 예측 레이블 표시
    color = (0, 255, 0)  # 기본 색상은 초록색
    if imagePred != adversaryPred:
        color = (0, 0, 255)  # 예측이 다르면 빨간색으로 변경

    cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    # 원본 이미지와 적대적 이미지를 수평으로 스택하여 비교
    output = np.hstack([image, adversary])
    cv2.imshow("FGSM Adversarial Images", output)
    cv2.waitKey(0)
