import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from keras.api.models import Sequential
from keras.api.layers import Dense, BatchNormalization, Activation, Dropout, Input
from keras import backend
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(73)


def create_fnn_model(hidden_layers, hidden_neurons, dropout_rate):

    backend.clear_session()
    fnn = Sequential()
    fnn.add(Input(shape=(30, )))

    for j in range(0, hidden_layers):
        fnn.add(Dense(units=hidden_neurons))
        fnn.add(BatchNormalization())
        fnn.add(Activation('gelu'))
        fnn.add(Dropout(dropout_rate))

    fnn.add(Dense(units=8))
    fnn.add(Activation('softmax'))

    opt = Adam(learning_rate=0.001)
    fnn.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return fnn


data = pd.read_csv('feature_set.csv')
feature_data = data.iloc[:, :-1]
class_data = data.iloc[:, -1]

x = feature_data.values
y = class_data.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,
                                                    random_state=73, stratify=y)

x_test_final, x_val, y_test_final, y_val = train_test_split(x_test, y_test, test_size=0.5,
                                                            random_state=73, stratify=y_test)

feature_scaler = StandardScaler()
feature_scaler.fit(x_train)
x_train_scaled = feature_scaler.transform(x_train)
x_test_final_scaled = feature_scaler.transform(x_test_final)
x_val_scaled = feature_scaler.transform(x_val)
joblib.dump(feature_scaler, 'scaler.pkl')

param_grid_fnn = {
    'hidden_layers': [2, 3],
    'hidden_neurons': [32, 64, 128],
    'dropout_rate': [0.2, 0.3]
}

fnn_model = KerasClassifier(build_fn=create_fnn_model, hidden_layers=2, hidden_neurons=32, dropout_rate=0.2,
                            batch_size=32, optimizer='adam', verbose=0, shuffle=True, random_state=73,
                            epochs=150)

stratified_k_fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=73)
grid = GridSearchCV(estimator=fnn_model, param_grid=param_grid_fnn, scoring='accuracy', n_jobs=-1,
                    cv=stratified_k_fold, verbose=0)
grid.fit(x_train_scaled, y_train)

best_parameters = grid.best_params_
best_accuracy = grid.best_score_
print(f"Best accuracy during CV: {best_accuracy * 100:.2f}% with best parameters: {best_parameters}")

best_fnn_model = create_fnn_model(
    hidden_layers= best_parameters['hidden_layers'],
    hidden_neurons= best_parameters['hidden_neurons'],
    dropout_rate= best_parameters['dropout_rate']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = best_fnn_model.fit(x_train_scaled, y_train, epochs=150, batch_size=32, verbose=0, shuffle=True,
                             validation_data=(x_val_scaled, y_val), callbacks=[early_stopping])

ts_loss, ts_accuracy = best_fnn_model.evaluate(x_test_final_scaled, y_test_final)
tr_loss, tr_accuracy = best_fnn_model.evaluate(x_train_scaled, y_train)

print(f'Test loss: {ts_loss}')
print(f'Test accuracy: {ts_accuracy * 100:.2f}%')
print(f'Train Loss: {tr_loss}')
print(f'Train Accuracy: {tr_accuracy * 100:.2f}%')

y_pred = best_fnn_model.predict(x_test_final_scaled)
y_pred = y_pred.argmax(axis=1)

avg_precision = precision_score(y_test_final, y_pred, average='macro')
avg_recall = recall_score(y_test_final, y_pred, average='macro')
avg_f1 = f1_score(y_test_final, y_pred, average='macro')

print(f"Precision: {avg_precision}")
print(f"Recall: {avg_recall}")
print(f"F1 Score: {avg_f1}")

cm = confusion_matrix(y_test_final, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))  # (width, height)
disp.plot(ax=ax)

# Compute precision and recall for each class
precision = precision_score(y_test_final, y_pred, average=None)
recall = recall_score(y_test_final, y_pred, average=None)


for i in range(cm.shape[0]):
    ax.text(-1.5, i, f"R: {recall[i]*100:.2f}%",  # Recall in percentage
            horizontalalignment='center', verticalalignment='center', color='black', fontsize=7, rotation=90)

for i, col_precision in enumerate(precision):
    ax.text(i, cm.shape[0] + 0.5, f"P: {col_precision*100:.2f}%",  # Precision in percentage
            horizontalalignment='center', verticalalignment='center', color='black', fontsize=7)

plt.tight_layout(pad=1.0)
plt.show()

best_fnn_model.save('final_model.h5')
best_fnn_model.export('final_model')

converter = tf.lite.TFLiteConverter.from_saved_model('final_model')
final_tflite_model = converter.convert()

with open("final_model.tflite", "wb") as f:
    f.write(final_tflite_model)
