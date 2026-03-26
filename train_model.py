from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Flatten, Dense# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau# type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Dataset directory
train_dir = r"C:\Users\laksh\OneDrive\Desktop\ASD\dataset\train"
val_dir = r"C:\Users\laksh\OneDrive\Desktop\ASD\dataset\valid"

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='binary')

# Callbacks
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3)

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[early_stop, reduce_lr])

# Save the model
model.save(r"C:\Users\laksh\OneDrive\Desktop\ASD\model\asdmodel.keras")