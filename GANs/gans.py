import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization,Input,Dense,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys,os

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
latten_dims=100
N,H,W=x_train.shape
D=H*W
x_train=x_train.reshape(-1,D)
y_train=y_train.reshape(N)

def build_generator(latten_dims):
    i=Input(shape=(latten_dims,))
    x=Dense(256,activation=LeakyReLU(alpha=0.2))(i)
    x=BatchNormalization(momentum=0.8)(x)
    x=Dense(512,activation=LeakyReLU(alpha=0.2))(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=Dense(1024,activation=LeakyReLU(alpha=0.2))(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=Dense(D,activation='tanh')(x)
    model=Model(i,x)
    return model


def build_discriminator(img_size):
    i=Input(shape=(img_size,))
    x=Dense(512,activation=LeakyReLU(alpha=0.2))(i)
    x=Dense(256,activation=LeakyReLU(alpha=0.2))(x)
    x=Dense(1,activation='sigmoid')(x)
    model=Model(i,x)
    return model

#make the combine model in which discriminator is freezed
discriminator=build_discriminator(D)
discriminator.compile(optimizer=Adam(0.0002,0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

generator=build_generator(latten_dims)
z=Input(shape=(latten_dims,))
img=generator(z)

#for freezing discriminator 
discriminator.trainable=False

fake_pred=discriminator(img)

combined_model=Model(z,fake_pred)
combined_model.compile(optimizer=Adam(0.0002,0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
print(fake_pred)
print(img)#this is output when we input noise in generator function
print(generator)#this is function

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')


batch_size=32
zeros=np.zeros(batch_size)
ones=np.ones(batch_size)
epochs=30000

def sample_images(epoch):
    rows,cols=5,5
    noise=np.random.randn(rows*cols,latten_dims)
    fake_img=generator.predict(noise)

    fake_img=(0.5*fake_img)+0.5
    fig,axs=plt.subplots(rows,cols)
    idx=0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(fake_img[idx].reshape(H,W),cmap='gray')
            axs[i,j].axis('off')
            idx =idx+1
    fig.savefig("gan_images/%d.png"%epoch)
    plt.close()


d_losses=[]
g_losses=[]
for epoch in range(epochs):
    #1st> Train the discriminator
    idx=np.random.randint(0,x_train.shape[0],batch_size)
    real_img=x_train[idx]

    noise=np.random.randn(batch_size,latten_dims)
    fake_img=generator.predict(noise)

    d_loss_real,d_acc_real=discriminator.train_on_batch(real_img,ones)
    d_loss_fake,d_acc_fake=discriminator.train_on_batch(fake_img,zeros)
    d_loss=0.5*(d_loss_real+d_loss_fake)
    d_acc=0.5*(d_acc_real+d_acc_fake)

    #2nd train generator
    noise=np.random.randn(batch_size,latten_dims)
    g_loss=combined_model.train_on_batch(noise,ones)

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch%200==0 :
        sample_images(epoch)

    if epoch%100==0:
        epoch=epoch+1
        print("epoch:{}/{}, d_loss:{}, d_acc:{}, g_loss:{}".format(epoch,epochs,d_loss,d_acc,g_loss))



