import os

import numpy as np

import keras as k
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(CSV_FILE_PATH):
    IRIS = pd.read_csv(CSV_FILE_PATH)
    target_var = 'Species'
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target,将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(IRIS['target'])
    y_bin_labels = []     # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        IRIS['y'+str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集
    print(IRIS)
    train_x, test_x, train_y, test_y = train_test_split(IRIS[features], IRIS[y_bin_labels],
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict

def main():
    # 0. 开始
    print("\nEra_DNN dataset using Keras/Tensorflow")
    np.random.seed(4)
    tf.random.set_seed(13)

    print("Load Iris data into memory")
    CSV_FILE_PATH = 'iris.csv'
    train_x, test_x, train_y, test_y, Class_dict = load_data(CSV_FILE_PATH)
    train_x = pd.DataFrame(train_x)
    train_x.astype(float)
    test_x = pd.DataFrame(test_x)
    test_x.astype(float)
    train_y = pd.DataFrame(train_y)
    train_y.astype(float)
    test_y = pd.DataFrame(test_y)
    test_y.astype(float)



    # 2. 定义模型
    init = k.initializers.glorot_uniform(seed=1)
    simple_adam = k.optimizers.Adam()
    model = k.models.Sequential()
    model.add(k.layers.Dense(units=5, input_dim=4, kernel_initializer=init, activation='relu'))
    model.add(k.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    #model.add(k.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(k.layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    # 3. 训练模型
    b_size = 1
    max_epochs = 100
    print("Starting training")


    tb_cb = k.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    es_cb = k.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.09, patience=5, verbose=0, mode='auto')
    cbks = [];
    cbks.append(tb_cb)
    cbks.append(es_cb)




    h = model.fit(train_x, train_y, callbacks= cbks, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=0, validation_data=(test_x, test_y))
    print("Training finished \n")

    # 4. 评估模型
    eval = model.evaluate(test_x, test_y, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
          % (eval[0], eval[1] * 100))

    # 5. 使用模型进行预测
    np.set_printoptions(precision=4)

    #6. 设定数采频率，每隔0.25秒扫描一次


    unknow = np.array([[4.8, 3, 1.1, 0.1]], dtype=np.float32)
    predicted = model.predict(unknow)
    print("Using model to predict species for features: ")
    print(unknow)
    print("\nPredicted softmax vector is: ")
    print(predicted)
    species_dict = {v:k for k, v in Class_dict.items()}
    print("\nPredicted species is: ")
    print(species_dict[np.argmax(predicted)])

if (__name__ == '__main__'):
    main()
