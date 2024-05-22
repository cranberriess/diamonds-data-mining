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
Построим матрицу рассеяния для категориальных признаков и численных, чтобы узнать взаимосвязь признаков между собой. Из графиков можно сделать вывод, что они между собой не связаны. Есть данные низкой, высокой и средней цены при разных признаках, поэтому определённую зависимость выявить трудно.<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/5542cea7-d3e7-4aa4-9180-6771e90bbe5f)<br>
Матрица рассеяния для численных признаков показывает, что цена зависит от веса бриллианта в каратах и в некоторой степени – от длины х.<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/54ec0d7d-8da3-44a0-a694-cb8966677864)<br>
Построим гистограмму для того, чтобы узнать частоту появления каждого признака: наибольшее количество бриллиантов имеют вес до 2 карат, среднюю общую глубину, цену до 5k.<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/d76f8e9b-f898-4856-8a47-2b41030e66a1)<br>

По круговой диаграмме и гистограмме можно сказать, что качество огранки у подавляющего большинства идеальное, премиум и очень хорошее качество приблизительно равны, а бриллиантов удовлетворительного качества совсем мало.<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/af75efab-e8b9-4399-a43a-b949bc1a3f13)<br>
Наибольшее количество бриллиантов имеет цвет G (почти бесцветный), чуть меньше – E (бесцветные) и F (с едва уловимым оттенком), наименьшее – J (с ясно выделенным оттенком).<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/8dbb3e5a-a693-40c4-8e3f-3903401ead33)<br>
По чистоте преобладают бриллианты класса SI1(мелкие заметные включения) и VS2(очень мелкие незначительные включения), меньше всего – I1(включения, заметные невооруженным глазом).<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/c6ddefb9-ac2d-412d-9659-a74e7af4fbcb)<br>

## Произведенная предварительная обработка:
- Удаление столбца индекса с помощью кода;
- Разделение признаков на категориальные и числовые;
- Преобразовали категориальные признаки;
- Масштабирование данных для дальнейшей работы;
- Проверка данных на наличие пропущенных значений.
## Подготовка данных к построению нейронных сетей:
В данном наборе данных будем предсказывать цену бриллианта, соответственно её мы выделим в переменную y и удалим из набора данных. Импортируем библиотеки, необходимые для подготовки данных к построению сетей и разделим набор данных на тестовую и обучающие выборки: 
```
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_2, y, test_size=0.33, random_state=42)
```
Установим Keras и Tensorflow, а также импортируем библиотеки для создания нейронной сети и создадим переменные обучения тензора:
```
pip install -U keras-tuner
%tensorflow_version 2.x
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
x_train_tensor = tf.constant(x_train)
y_train_tensor = tf.constant(y_train)
```
## Нейронные сети и их архитектуры: 
Создадим нейронную сеть, добавим слои Dense с разными весами и функцией активации «relu», причём укажем регуляризатор L1
```
model = tf.keras.Sequential([tf.keras.layers.Dense(16,input_shape=(9,),activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                          tf.keras.layers.Dense(32,activation='relu'),
                           tf.keras.layers.Dense(32,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(32,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(13,activation='relu',activity_regularizer=tf.kers.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(16,activation='relu',activity_regularizer=tf.keras.regularizers.L1(0.01)),
                           tf.keras.layers.Dense(1,activation=None)])
```
Запустим компиляцию с оптимизатором Adam и измерим качество работы расчётом среднеквадратической ошибки mse:
```
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.mse)
model.summary()
```
Запустим обучение сети на 100 эпохах, размером партии 64 и частью проверочных данных размером 0.2:
```
history = model.fit(x_train_tensor,y_train_tensor,batch_size=64,
epochs=100,validation_split = 0.2)
```
Загрузим предсказанные данные в переменную pred и сравним полученные значения и правильные значения:
```
pred = model.predict(x_test)
vec = np.array([])
for i in np.arange(0, len(pred)):
  vec = np.append(vec, pred[i][0])
for i in np.arange(0, len(y_test)):
    print("Предсказанное значение:", pred[i][0], ", правильное значение:", y_test[i], 'разница: ', np.abs(pred[i][0] - y_test[i]))
```

На основе этих данных посчитаем корреляцию с истинными данными:
```
СС_tuner = np.corrcoef(vec, y_test)
СС_tuner = СС_tuner[0][1]
print(f'Коэффициент корреляции с истинными данными: {СС_tuner}')
```
Коэффициент корреляции с истинными данными: 0.9881446388405455<br>
Вывод: Коэффициент корреляции близок к единице, это говорит о том, что предсказанные данные близки к истинным.
## Эксперименты с нейронной сетью:
1)	В качестве эксперимента изменим архитектуру сети, убрав из неё регуляризатор и несколько слоёв, оставим веса равные 8 и добавим слой Dropout, все остальные значения оставим прежними:
```
model_2 = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(9,),activation='relu'),
                     tf.keras.layers.Dropout(.2, input_shape=(9,)),
                     tf.keras.layers.Dense(8,activation='relu'),
                     tf.keras.layers.Dense(1,activation=None)])
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.mse)
model_2.summary()
history_2 = model_2.fit(x_train_tensor,y_train_tensor,batch_size=64,epochs=100,validation_split = 0.2)
```
Коэффициент корреляции с истинными данными: 0.9640900301079525<br>
Вывод: в сравнении с первой нейронной сетью предсказанные данные второй сетью менее точные, больше ошибка и ниже корреляция. Значит, сеть обучена хуже, чем первая.
2)	В архитектуре нейронной сети изменим функцию активации на «tanh»:
```
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
history_3 = model_3.fit(x_train_tensor,y_train_tensor,batch_size=64,epochs=100,validation_split = 0.2)
```
Коэффициент корреляции с истинными данными: 0.9685227711244739<br>
Вывод: предсказанные данные чуть более точные чем данные, предсказанные второй сетью, но не такие точные, как данные после обучения первой сети.
## График изменения качества нейронной сети от ее архитектуры:
Отразим изменение качества нейронной сети на графике потерь и эпох обучения, где выведем кривые обучения и проверки:
1)	График для нейронной сети с регуляризатором L1, наибольшие веса, функция активация Relu:<br>
 ![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/7d160f6a-fe99-496f-97f1-71ffd992ce04)<br>
2)	График для нейронной сети без регуляризатора, наименьшие веса, функция активация Relu:<br>
![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/c7ba30bd-cb5c-4c56-b700-7ae4274db2e5)<br>
3)	График для нейронной сети с регуляризатором L1, наибольшие веса, функция активация tanh:<br>
 ![image](https://github.com/cranberriess/diamonds-data-mining/assets/105839329/53e6662a-bb8b-4cee-a851-a656db8de0be)<br>
## Результаты
Качество обучения нейронной сети напрямую зависит от его архитектуры. Для данного набора наиболее подходит нейронная сеть с указанным регуляризатором L1 и функции активации Relu. Нейронная сеть с таким же регуляризатором, но функцией активации tanh справилась хуже, сеть с меньшим количеством слоёв и без указания регуляризатора предсказала самые неточные данные. Это видно на графике: кривая валидации выше кривой обучения, значит, сеть является недообученной.
