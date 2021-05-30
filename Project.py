#FRONT END
from tkinter import *
from tkinter import ttk 
from tkinter import filedialog
from PIL import Image,ImageTk
from keras.models import load_model  
from keras.preprocessing import image
import numpy as np

class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("tkinter dialog")
        self.minsize(800,800)
        self.labelFrame=ttk.LabelFrame(self,text="open a file")
        self.labelFrame.grid(column=0,row=1,padx=20,pady=20)
        self.button() 
    def button(self):
        self.button=ttk.Button(self.labelFrame,text="browse a file",command=self.filedialog)
        self.evaluate = ttk.Button(self.labelFrame,text ="evaluate",command = self.getvalue)
        self.button.grid(sticky=N+E+W+S)
        self.evaluate.grid(sticky=N+E+W+S)
    def filedialog(self):
        self.filename=filedialog.askopenfilename(initialdir="/",title="select a file",filetype=(("jpeg",".jpg"),("all files","*.*"))) 
        self.img=ImageTk.PhotoImage(Image.open(self.filename))  
        self.label=ttk.Label(self.labelFrame,text=self.filename,image=self.img)
        self.label.grid(sticky=N+E+W+S)
        self.label.configure(text=self.filename)  
    def getvalue(self):
        new_model=load_model('cat_dog_100epochs.h5') 
dog_img=image.load_img(self.filename,target_size=(150,150))
        dog_img=image.img_to_array(dog_img)
        dog_img=np.expand_dims(dog_img,axis=0)
        dog_img=dog_img/255
        a=new_model.predict(dog_img)
        b=new_model.predict_classes(dog_img)
        self.label=ttk.Label(self.labelFrame,text=f'probability is {a}')
        self.label.grid(sticky=N+E+W+S)
        self.label.configure(text=f'probability is {a}')
        self.label=ttk.Label(self.labelFrame,text=f'class of image is {b}')
        self.label.grid(sticky=N+E+W+S)
        self.label.configure(text=f'class of image is {b}')
if __name__=='__main__':
    root=Root()
    root.mainloop()

#BACK END
import matplotlib.pyplot as plt
import cv2
# Technically not necessary in newest versions of jupyter
%matplotlib inline
cat4 = cv2.imread('CATS_DOGS/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4,cv2.COLOR_BGR2RGB)
type(cat4)
cat4.shape
plt.imshow(cat4)
 
dog2 = cv2.imread('CATS_DOGS/train/Dog/2.jpg')
dog2 = cv2.cvtColor(dog2,cv2.COLOR_BGR2RGB)
dog2.shape
plt.imshow(dog2)
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
plt.imshow(image_gen.random_transform(dog2))
image_gen.flow_from_directory('CATS_DOGS/train')
image_gen.flow_from_directory('CATS_DOGS/test')
image_shape = (150,150,3)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))
# Last layer, remember its binary, 0=cat , 1=dog
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16
train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.class_indices
from IPython.display import display
from PIL import Image
results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=100, 
                              validation_data=test_image_gen,
                             validation_steps=12)
model.save('cat_dogclassification.h5')
results.history['acc']
plt.plot(results.history['acc'])
model.save('cat_dog_100epochs.h5')
train_image_gen.class_indices
import numpy as np
from keras.preprocessing import image
dog_file = 'CATS_DOGS/train/Cat/2.jpg'
dog_img = image.load_img(dog_file, target_size=(150, 150))
dog_img = image.img_to_array(dog_img)
dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img/255
prediction_prob = model.predict(dog_img)
print(f'Probability that image is a dog is: {prediction_prob} ')
from keras.models import load_model  
new_model=load_model('cat_dogclassification.h5')
dog_file='CATS_DOGS/test/DOG/9380.jpg'
from keras.preprocessing import image
dog_img=image.load_img(dog_file,target_size=(150,150))
dog_img=image.img_to_array(dog_img)
import numpy as np
dog_img=np.expand_dims(dog_img,axis=0)
dog_img=dog_img/255
new_model.predict(dog_img)
new_model.predict_classes(dog_img)
