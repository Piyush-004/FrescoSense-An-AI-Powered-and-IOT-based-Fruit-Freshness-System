# Step 1: Import Libraries
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 2: Prepare Dataset
train_dir = 'Train'
validation_dir = 'Test'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Change to 'binary' if it's a binary classification

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # Change to 'binary' if it's a binary classification

# Generate label file
class_labels = sorted(train_generator.class_indices.keys())
with open('lb.txt', 'w') as f:
    for label in class_labels:
        f.write(label + '\n')

# Step 3: Build Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_labels), activation='softmax')  # Output layer with softmax activation
])

# Step 4: Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change to 'binary_crossentropy' if binary classification
              metrics=['accuracy'])

# Step 5: Train Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=500,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Step 6: Evaluate Model
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print('Test accuracy:', test_acc)

# Step 7: Save Model
model.save('image_classification_model.h5')
