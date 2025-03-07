import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import datetime

DATASET_DIR = 'dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 20
MODEL_SAVE_PATH = 'model_car_brand.h5'
CLASS_INDICES_PATH = 'class_indices.json'


def convert_images_to_rgb(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(path)
                except Exception as e:
                    print(f"Nie można przetworzyć {path}: {e}")


def prepare_data(dataset_dir, image_size):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


def build_model(image_size, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3),
                          name='resnet50_base')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def train_model(model, train_generator, validation_generator, epochs):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[reduce_lr, tensorboard_callback, early_stopping]
    )

    return history


def fine_tune_model(model, base_model, train_generator, validation_generator, fine_tune_epochs=10):
    base_model.trainable = True

    fine_tune_at = -50
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    log_dir = "logs/fine_tune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history_fine = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        callbacks=[reduce_lr, tensorboard_callback, early_stopping]
    )

    return history_fine


def plot_confusion_matrix(model, validation_generator):
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=validation_generator.class_indices.keys(),
                yticklabels=validation_generator.class_indices.keys(),
                cmap='Blues')
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidywana klasa')
    plt.title('Macierz pomyłek')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))


def plot_training_history(history, history_fine=None):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)

    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], label='Strata treningowa')
    plt.plot(epochs, history.history['val_loss'], label='Strata walidacyjna')

    if history_fine:
        # Epoki dla fine-tuningu
        fine_epochs = range(len(history.history['loss']) + 1,
                            len(history.history['loss']) + 1 + len(history_fine.history['loss']))
        plt.plot(fine_epochs, history_fine.history['loss'], label='Strata treningowa (fine-tune)', linestyle='--')
        plt.plot(fine_epochs, history_fine.history['val_loss'], label='Strata walidacyjna (fine-tune)', linestyle='--')

    plt.title('Strata podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(epochs, history.history['accuracy'], label='Dokładność treningowa')
    plt.plot(epochs, history.history['val_accuracy'], label='Dokładność walidacyjna')

    if history_fine:
        plt.plot(fine_epochs, history_fine.history['accuracy'], label='Dokładność treningowa (fine-tune)',
                 linestyle='--')
        plt.plot(fine_epochs, history_fine.history['val_accuracy'], label='Dokładność walidacyjna (fine-tune)',
                 linestyle='--')

    plt.title('Dokładność podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_brand(model, class_indices, image_path, image_size):
    idx_to_class = {v: k for k, v in class_indices.items()}

    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_brand = idx_to_class[predicted_class]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Przewidywana marka: {predicted_brand}')
    plt.show()

    return predicted_brand


def save_class_indices(class_indices, filepath):
    with open(filepath, 'w') as f:
        json.dump(class_indices, f)


def load_class_indices(filepath):
    with open(filepath, 'r') as f:
        class_indices = json.load(f)
    return class_indices


def main():
    print("Konwersja obrazów do RGB...")
    convert_images_to_rgb(DATASET_DIR)

    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_INDICES_PATH):
        print(
            f"Znaleziono zapisany model '{MODEL_SAVE_PATH}' oraz mapowanie klas '{CLASS_INDICES_PATH}'. Ładowanie modelu...")
        model = load_model(MODEL_SAVE_PATH)
        class_indices = load_class_indices(CLASS_INDICES_PATH)
    else:
        print("Zapisany model lub mapowanie klas nie zostało znalezione. Rozpoczynanie treningu...")

        print("Przygotowanie danych...")
        train_gen, val_gen = prepare_data(DATASET_DIR, IMAGE_SIZE)
        num_classes = len(train_gen.class_indices)
        print(f'Liczba klas: {num_classes}')
        print(f'Nazwa klas: {train_gen.class_indices}')

        print("Budowanie modelu...")
        model, base_model = build_model(IMAGE_SIZE, num_classes)
        model.summary()

        print("Trenowanie modelu...")
        history = train_model(model, train_gen, val_gen, EPOCHS)

        print("Fine-tuning modelu...")
        history_fine = fine_tune_model(model, base_model, train_gen, val_gen, FINE_TUNE_EPOCHS)

        model.save(MODEL_SAVE_PATH)
        print(f'Model zapisany jako {MODEL_SAVE_PATH}')

        class_indices = train_gen.class_indices
        save_class_indices(class_indices, CLASS_INDICES_PATH)
        print(f'Mapowanie klas zapisane jako {CLASS_INDICES_PATH}')

        print("Generowanie wykresów procesu uczenia się...")
        plot_training_history(history, history_fine)

    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_INDICES_PATH):
        print("Przygotowanie danych do ewaluacji...")
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     validation_split=0.2)
        val_gen = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    else:
        print("Przygotowanie danych do ewaluacji...")
        train_gen, val_gen = prepare_data(DATASET_DIR, IMAGE_SIZE)

    if os.path.exists(MODEL_SAVE_PATH):
        print("Generowanie macierzy pomyłek...")
        plot_confusion_matrix(model, val_gen)

    while True:
        user_input = input("Podaj ścieżkę do obrazka lub 'exit' aby zakończyć: ")
        if user_input.lower() == 'exit':
            break
        if os.path.exists(user_input):
            try:
                if os.path.exists(CLASS_INDICES_PATH):
                    class_indices = load_class_indices(CLASS_INDICES_PATH)
                else:
                    print(f"Mapowanie klas '{CLASS_INDICES_PATH}' nie zostało znalezione.")
                    continue

                predict_brand(model, class_indices, user_input, IMAGE_SIZE)
            except Exception as e:
                print(f"Wystąpił błąd podczas przewidywania: {e}")
        else:
            print("Ścieżka nie istnieje. Spróbuj ponownie.")


if __name__ == "__main__":
    main()
