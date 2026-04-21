# -------------------- IMPORTS --------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# -------------------- PATHS --------------------
train_dir = 'DataSet/Training'
val_dir   = 'DataSet/Testing'

# -------------------- PARAMETERS --------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 40
EPOCHS_FINETUNE= 20 
FINE_TUNE_LAYERS = 40 #to unfreeze last 40 layers of model


# -------------------- DATA GENERATORS --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode= 'nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle= False
)

# -------------------- MODEL SETUP --------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------- FREEZE BASE MODEL --------------------
for layer in base_model.layers:
    layer.trainable = False

# -------------------- COMPILE --------------------
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- CALLBACKS --------------------
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_mobilenetv2.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# -------------------- TRAIN --------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    callbacks=[es, checkpoint]
)

print("✅ Training finished. Best model saved as best_mobilenetv2.h5")
# -------------------- FINE-TUNING LAST LAYERS --------------------

for layer in base_model.layers[-FINE_TUNE_LAYERS:]: # last 40 layers
    layer.trainable=True

#Lowering learning rate for fine-tuning
model.compile(
    optimizer= Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_finetune= model.fit(
    train_generator,
    validation_data=val_generator,
    epochs= EPOCHS_FINETUNE,
    callbacks=[es, checkpoint]
)
print("✅ Fine-tuning finished. Best model saved as best_mobilenetv2.h5")
# -------------------- VISUALIZE --------------------
plt.figure(figsize=(8,5))
plt.plot(history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# -------------------- SAVE MODEL --------------------
model.save('trial_model.h5') #--changed from final_mobilenetv2.py
print("🎉 Final model saved as trial_model.h5")
