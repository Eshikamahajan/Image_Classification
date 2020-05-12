# Convolutional Neural Network
#WE have manually done the preprocessing by saving the images as test and train separately.

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential #using sequential method to build the neural network
from keras.layers import Convolution2D #dealing with 2D imamges and thus Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense #to add fully connected layers

# Initialising the CNN
classifier = Sequential() # creatinf obj of Sequential class

# Step 1 - Convolution 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
'''
num_featuremaps=32,num_rows_kernel=3,num_col_kernel=3 kernel==filter
we also have border_mode which is eual to 'same' here which determines the method of padding the
bpundary pirxels. by default it takes the same value therfore we ignore mentioning it explicitly
input_shape is the expected shape in which we would want to convert our images beore trainnig the model
64x64 is the size of the image and 3 corresponds to the RGB value, it will be 1 if using black and white images
so this creates 32 feature maps of 3x3 size we created 32 bcoz we are working on CPU so we wil
slowly increase the no of feature map by adding more convolution layers
we are using relu activation function to reduce the possibilty of negative pixels while convolution
bcoz we dont want them in our images. thus increasing the non linearity of the model
'''
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#pool_size is the size of the sliding window

# Adding a second convolutional layer to increase the accuracy and to decrease the difference between test and training accuracy
#we donot need to mention the input_shape again bcoz the network already knows that this is the second convoltuion later added

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection through Dense
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#final output layer we use sigmoid to convert the output as binary categories
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''
optimizer --> we choose the stoch. grad. descent adam algo
loss--> binary bcoz we have 2 outputs
        cross_entropy bcoz we are dealing withlogrithmic loss
        if we had more than 2 classes than we would have 
        choosen categorical_crossentropy
we determine the accuracy of the model using accuracy metrics.
'''

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#shifting etc is done to prevent overfitting as it will change the inclination etc of the image and 
#our model will get trained on same yet geometrically different images.(IMAGE AUGMENTATION)
#rescaking to get all the values between 0 and 1
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
#smaples_per_epoch== total no of samples we have.
#nb_val_samples how much test samples we have
'''
To increase teh accuracy of the model you should take more pixels into account 
here we took 64x64 but you may take 128x128 to enhance the features the model is fed on

'''