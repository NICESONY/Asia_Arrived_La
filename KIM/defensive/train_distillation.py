# https://github.com/lengstrom/defensive-distillation 참고
# teacher 모델의 soft labels를 이용하여 student 모델을 학습

import numpy as np
import tensorflow as tf
import python.defensive.train_baseline as train_baseline
from python.defensive.model import make_model
from python.defensive.setup import *

def train(train_data, train_labels, teacher_name, file_name,
          NUM_EPOCHS=50, BATCH_SIZE=128, TRAIN_TEMP=1):
    # Step 1: train the teacher model
    # train_baseline.train 함수는 train_data, train_labels를 이용해 teacher model train
    train_baseline.train(train_data, train_labels, teacher_name,
                         NUM_EPOCHS, BATCH_SIZE, TRAIN_TEMP)
    # TRAIN_TEMP : 온도 매개변수 -> 모델의 softmax 출력 조절

    # Step 2: evaluate the model on the training data at the training temperature
    soft_labels = np.zeros(train_labels.shape)
    # 학습된 teacher 모델을 사용해 원본 train_data에 대해 예측 수행, 그 결과를 sofr_labels에 저장
    with tf.Session() as s:
        model = make_model(teacher_name)
        xs = tf.placeholder(tf.float32,
                            shape=(100, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        predictions = tf.nn.softmax(model(xs)/TRAIN_TEMP)
        # softmax 출력 계산 시 TRAIN_TEMP 사용. 온도가 높을수록 출력을 부드럽게(클래스 간의 확률 분포를 넓게) 만들어 student 모델이 더 일반화된 학습을 하게 도움
        
        for i in range(0,len(train_data),100):
            predicted = predictions.eval({xs: train_data[i:i+100]})
            soft_labels[i:i+100] = predicted

    # Step 3: train the distilled model on the new training data
    # 생성된 soft label를 사용해 student 모델 학습(teacher 모델에서 얻은 지식을 전달하는 것이 목적)
    train_baseline.train(train_data, soft_labels, file_name,
                         NUM_EPOCHS, BATCH_SIZE, TRAIN_TEMP)
    
        
teacher_name = "models/teacher"
file_name = "models/distilled"
train(train_data, train_labels, teacher_name, file_name,
      NUM_EPOCHS=50, BATCH_SIZE=128, TRAIN_TEMP=100)
