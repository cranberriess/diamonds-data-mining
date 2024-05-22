# diamonds-data-mining

## Краткое описание данных: diamonds - фрейм данных с 53940 строками и 10 переменными: 
•	цена в долларах США (\$326--\$18 823);
•	вес бриллианта в каратах (0,2--5,01);
•	качество огранки (Удовлетворительно, Хорошее, Очень хорошее, Премиум, Идеальное);
•	цвет бриллианта, от J (худший) до D (лучший); 
•	чистота - показатель чистоты алмаза (I1 (худший), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (лучший));
•	длина x в мм (0--10,74);
•	ширина y в мм (0--58,9);
•	глубина z в мм (0--31,8);
•	глубина - общая глубина в процентах = z / среднее значение (x, y) = 2 * z / (x + y) (43--79);
•	таблица ширины вершины ромба относительно самой широкой точки (43--95).
## Графики и их смысл:
Построим матрицу рассеяния для категориальных признаков и численных, чтобы узнать взаимосвязь признаков между собой. Из графиков можно сделать вывод, что они между собой не связаны. Есть данные низкой, высокой и средней цены при разных признаках, поэтому определённую зависимость выявить трудно.
import plotly.express as px
fig = px.scatter_matrix(data, dimensions=["cut",  "color", "clarity", "price"])
fig.show()
 
Матрица рассеяния для численных признаков показывает, что цена зависит от веса бриллианта в каратах и в некоторой степени – от длины х.
fig = px.scatter_matrix(data, dimensions=["carat", "depth", "table", "x", "y",  "z", "price"])
fig.show() 
Построим гистограмму для того, чтобы узнать частоту появления каждого признака: наибольшее количество бриллиантов имеют вес до 2 карат, среднюю общую глубину, цену до 5k.
#построение гистограммы (распределения частот) для каждого признака
numeric_features.hist(bins = 10, rwidth= .9, sharey= True)
 
По круговой диаграмме и гистограмме можно сказать, что качество огранки у подавляющего большинства идеальное, премиум и очень хорошее качество приблизительно равны, а бриллиантов удовлетворительного качества совсем мало.

Наибольшее количество бриллиантов имеет цвет G (почти бесцветный), чуть меньше – E (бесцветные) и F (с едва уловимым оттенком), наименьшее – J (с ясно выделенным оттенком).
  
По чистоте преобладают бриллианты класса SI1(мелкие заметные включения) и VS2(очень мелкие незначительные включения), меньше всего – I1(включения, заметные невооруженным глазом).
  
 
## Произведенная предварительная обработка:
Удалили столбец индекса с помощью кода:
data = data.drop(columns=['Unnamed: 0'])
Разделили признаки на категориальные и числовые, преобразовали категориальные признаки:
categorical_features = data[['cut', 'color', 'clarity']].copy()
numeric_features = data[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']].copy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categorial_encoded = categorical_features.apply(le.fit_transform)
categorial_encoded.reset_index(drop=True, inplace=True)
numeric_features.reset_index(drop=True, inplace=True)
data_2 = pd.concat([categorial_encoded, numeric_features], axis = 1)
Отмасштабировали данные для дальнейшей работы:
mymean = data_2.mean(axis=0)
mystd = data_2.std(axis=0)
data_2 -= mymean
data_2 /= mystd
data_2
Проверили данные на наличие пропущенных значений:
print(data.isnull().values.any())
False
## Подготовка данных к построению нейронных сетей:
В данном наборе данных будем предсказывать цену бриллианта, соответственно её мы выделим в переменную y и удалим из набора данных.
y = data_2['price']
del data_2['price']
Импортируем библиотеки, необходимые для подготовки данных к построению сетей и разделим набор данных на тестовую и обучающие выборки:
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_2, y, test_size=0.33, random_state=42)
Установим Keras и Tensorflow, а также импортируем библиотеки для создания нейронной сети и создадим переменные обучения тензора:
pip install -U keras-tuner
%tensorflow_version 2.x
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
x_train_tensor = tf.constant(x_train)
y_train_tensor = tf.constant(y_train)
## Нейронные сети и их архитектуры: 
Создадим нейронную сеть, добавим слои Dense с разными весами и функцией активации «relu», причём укажем регуляризатор L1
model = tf.keras.Sequential([tf.keras.layers.Dense(16,input_shape=(9,),activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                          tf.keras.layers.Dense(32,activation='relu'),
                           tf.keras.layers.Dense(32,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(32,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(13,activation='relu',activity_regularizer=tf.kers.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(1,activation=None)])
Запустим компиляцию с оптимизатором Adam и измерим качество работы расчётом среднеквадратической ошибки mse:
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.mse)
model.summary()
output
Model: "sequential_3"
__________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_19 (Dense)            (None, 16)                160         
 dense_20 (Dense)            (None, 32)                544                                                                       
 dense_21 (Dense)            (None, 32)                1056                                                                       
 dense_22 (Dense)            (None, 32)                1056                                                                     
 dense_23 (Dense)            (None, 13)                429                                                                      
 dense_24 (Dense)            (None, 16)                224                                                                      
 dense_25 (Dense)            (None, 16)                272                                                                       
 dense_26 (Dense)            (None, 1)                 17                                                                        
=================================================================
Total params: 3758 (14.68 KB)
Trainable params: 3758 (14.68 KB)
Non-trainable params: 0 (0.00 Byte)
Запустим обучение сети на 100 эпохах, размером партии 64 и частью проверочных данных размером 0.2:
history = model.fit(x_train_tensor,y_train_tensor,batch_size=64,
epochs=100,validation_split = 0.2)
output
Epoch 95/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0327 - val_loss: 0.0344
Epoch 96/100
452/452 [==============================] - 2s 4ms/step - loss: 0.0325 - val_loss: 0.0292
Epoch 97/100
452/452 [==============================] - 2s 3ms/step - loss: 0.0325 - val_loss: 0.0307
Epoch 98/100
452/452 [==============================] - 2s 4ms/step - loss: 0.0319 - val_loss: 0.0330
Epoch 99/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0329 - val_loss: 0.0302
Epoch 100/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0321 
- val_loss: 0.0335
Загрузим предсказанные данные в переменную pred и сравним полученные значения и правильные значения:
pred = model.predict(x_test)
vec = np.array([])
for i in np.arange(0, len(pred)):
  vec = np.append(vec, pred[i][0])
for i in np.arange(0, len(y_test)):
    print("Предсказанное значение:", pred[i][0], ", правильное значение:", y_test[i], 'разница: ', np.abs(pred[i][0] - y_test[i]))
output
Предсказанное значение: -0.7817735 , правильное значение: -0.7519350883357471 разница:  0.029838419259315185
Предсказанное значение: -0.86168355 , правильное значение: -0.8862897935528367 разница:  0.024606246056041003
Предсказанное значение: -0.4046712 , правильное значение: -0.4173016341103466 разница:  0.012630441941157144
Предсказанное значение: -0.82737 , правильное значение: -0.8308935438271337 разница:  0.0035235558625035246
Предсказанное значение: -0.093798324 , правильное значение: -0.24008376734078993 разница:  0.14628544287503828
Предсказанное значение: -0.82259804 , правильное значение: -0.7782545734995426 разница:  0.04434346660436972
Предсказанное значение: -0.4516666 , правильное значение: -0.4807190602669206 разница:  0.02905246671528483
На основе этих данных посчитаем корреляцию с истинными данными:
СС_tuner = np.corrcoef(vec, y_test)
СС_tuner = СС_tuner[0][1]
print(f'Коэффициент корреляции с истинными данными: {СС_tuner}')
output
Коэффициент корреляции с истинными данными: 0.9881446388405455
Вывод: Коэффициент корреляции близок к единице, это говорит о том, что предсказанные данные близки к истинным.
## Эксперименты с нейронной сетью:
1)	В качестве эксперимента изменим архитектуру сети, убрав из неё регуляризатор и несколько слоёв, оставим веса равные 8 и добавим слой Dropout, все остальные значения оставим прежними:
model_2 = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(9,),activation='relu'),
                     tf.keras.layers.Dropout(.2, input_shape=(9,)),
                     tf.keras.layers.Dense(8,activation='relu'),
                     tf.keras.layers.Dense(1,activation=None)])
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.mse)
model_2.summary()
output
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_8 (Dense)             (None, 8)                 80                                                                         
 dropout (Dropout)           (None, 8)                 0                                                                        
 dense_9 (Dense)             (None, 8)                 72                                                                       
 dense_10 (Dense)            (None, 1)                 9                                                                          
=================================================================
Total params: 161 (644.00 Byte)
Trainable params: 161 (644.00 Byte)
Non-trainable params: 0 (0.00 Byte)
history_2 = model_2.fit(x_train_tensor,y_train_tensor,batch_size=64,epochs=100,validation_split = 0.2)
output
Epoch 95/100
452/452 [==============================] - 1s 2ms/step - loss: 0.0825 - val_loss: 0.1038
Epoch 96/100
452/452 [==============================] - 2s 3ms/step - loss: 0.0819 - val_loss: 0.1280
Epoch 97/100
452/452 [==============================] - 2s 4ms/step - loss: 0.0836 - val_loss: 0.1083
Epoch 98/100
452/452 [==============================] - 2s 3ms/step - loss: 0.0815 - val_loss: 0.1262
Epoch 99/100
452/452 [==============================] - 2s 4ms/step - loss: 0.0898 - val_loss: 0.1077
Epoch 100/100
452/452 [==============================] - 2s 3ms/step - loss: 0.0822 - val_loss: 0.1046
Уже в истории мы видим, что ошибка больше, чем в архитектуре первой сети. Измерим коэффициент корреляции:
for i in np.arange(0, len(y_test_2)):
    print("Предсказанное значение:", pred_2[i][0], ", правильное значение:", y_test_2[i], 'разница: ', np.abs(pred_2[i][0] - y_test_2[i]))
Предсказанное значение: -0.62995493 , правильное значение: -0.7519350883357471 разница:  0.12198015421556885
Предсказанное значение: -0.70870453 , правильное значение: -0.8862897935528367 разница:  0.17758526236005712
Предсказанное значение: -0.36586174 , правильное значение: -0.4173016341103466 разница:  0.051439890421763224
Предсказанное значение: -0.6910958 , правильное значение: -0.8308935438271337 разница:  0.13979771481712389
Предсказанное значение: -0.055559427 , правильное значение: -0.24008376734078993 разница:  0.18452434079469313

СС_tuner_2 = np.corrcoef(vec_2, y_test_2)
СС_tuner_2 = СС_tuner_2[0][1]
print(f'Коэффициент корреляции с истинными данными: {СС_tuner_2}')

Коэффициент корреляции с истинными данными: 0.9640900301079525
Вывод: в сравнении с первой нейронной сетью предсказанные данные второй сетью менее точные, больше ошибка и ниже корреляция. Значит, сеть обучена хуже, чем первая.
2)	В архитектуре нейронной сети изменим функцию активации на «tanh»:
model_3 = tf.keras.Sequential([tf.keras.layers.Dense(16,input_shape=(9,),activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                          tf.keras.layers.Dense(32,activation='tanh'),
                           tf.keras.layers.Dense(32,activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(32,activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(13,activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='tanh',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(1,activation=None)])
model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),    loss=tf.keras.losses.mse)
model_3.summary()
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_35 (Dense)            (None, 16)                160                                                                       
 dense_36 (Dense)            (None, 32)                544                                                                       
 dense_37 (Dense)            (None, 32)                1056                                                                      
 dense_38 (Dense)            (None, 32)                1056                                                                      
 dense_39 (Dense)            (None, 13)                429                                                                       
 dense_40 (Dense)            (None, 16)                224                                                                       
 dense_41 (Dense)            (None, 16)                272                                                                       
 dense_42 (Dense)            (None, 1)                 17                                                                        
=================================================================
Total params: 3758 (14.68 KB)
Trainable params: 3758 (14.68 KB)
Non-trainable params: 0 (0.00 Byte)
history_3 = model_3.fit(x_train_tensor,y_train_tensor,batch_size=64,epochs=100,validation_split = 0.2)
Epoch 95/100
452/452 [==============================] - 1s 3ms/step - loss: 0.0846 - val_loss: 0.0756
Epoch 96/100
452/452 [==============================] - 1s 3ms/step - loss: 0.0843 - val_loss: 0.0740
Epoch 97/100
452/452 [==============================] - 2s 4ms/step - loss: 0.0838 - val_loss: 0.0748
Epoch 98/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0839 - val_loss: 0.0930
Epoch 99/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0830 - val_loss: 0.0771
Epoch 100/100
452/452 [==============================] - 2s 5ms/step - loss: 0.0845 - val_loss: 0.0751

for i in np.arange(0, len(y_test_3)):
    print("Предсказанное значение:", pred_3[i][0], ", правильное значение:", y_test_3[i], 'разница: ', np.abs(pred_3[i][0] - y_test_3[i]))
output
Предсказанное значение: -0.76582754 , правильное значение: -0.7519350883357471 разница:  0.013892448247199707
Предсказанное значение: -0.7829993 , правильное значение: -0.8862897935528367 разница:  0.1032905164379685
Предсказанное значение: -0.36458552 , правильное значение: -0.4173016341103466 разница:  0.0527161152733715
Предсказанное значение: -0.6758474 , правильное значение: -0.8308935438271337 разница:  0.15504613267143297
Предсказанное значение: -0.13430455 , правильное значение: -0.24008376734078993 разница:  0.10577921407044996

СС_tuner_3 = np.corrcoef(vec_3, y_test_3)
СС_tuner_3 = СС_tuner_3[0][1]
print(f'Коэффициент корреляции с истинными данными: {СС_tuner_3}')

Коэффициент корреляции с истинными данными: 0.9685227711244739
## Вывод: предсказанные данные чуть более точные чем данные, предсказанные второй сетью, но не такие точные, как данные после обучения первой сети.
График изменения качества нейронной сети от ее архитектуры:
Отразим изменение качества нейронной сети на графике потерь и эпох обучения, где выведем кривые обучения и проверки:
1)	График для нейронной сети с регуляризатором L1, наибольшие веса, функция активация Relu:
 
 
 
2)	График для нейронной сети без регуляризатора, наименьшие веса, функция активация Relu:

 
 
3)	График для нейронной сети с регуляризатором L1, наибольшие веса, функция активация tanh:
 
 
 
## Результаты: качество обучения нейронной сети напрямую зависит от его архитектуры. Для данного набора наиболее подходит нейронная сеть с указанным регуляризатором L1 и функции активации Relu. Нейронная сеть с таким же регуляризатором, но функцией активации tanh справилась хуже, сеть с меньшим количеством слоёв и без указания регуляризатора предсказала самые неточные данные. Это видно на графике: кривая валидации выше кривой обучения, значит, сеть является недообученной.
