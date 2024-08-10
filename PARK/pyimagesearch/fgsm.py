from tensorflow.keras.losses import MSE
import tensorflow as tf
def generate_image_adversary(model, image, label, eps=2 / 255.0):
    image=tf.cast(image,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred=model(image)
        loss=MSE(label,pred)
        gradient=tape.gradient(loss,image)
        signedGrad=tf.sign(gradient)
        # if bTargeted: gradient*=-1  먹표하고 가까워져야 하기 떄문에 다시 미니머마이즈로 가야하기 떄문에 -1를 해준다
        adversary=(image+(signedGrad*eps)).numpy()
        return adversary       