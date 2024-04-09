import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb

from tqdm import tqdm
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------------------------

def read_omg_csv(path_palm_data: str, 
                 n_omg_channels: int = 50, 
                 n_acc_channels: int = 3, 
                 n_gyr_channels: int = 3, 
                 n_mag_channels: int = 6, 
                 n_enc_channels: int = 0,
                 button_ch: bool = True, 
                 sync_ch: bool = True, 
                 timestamp_ch: bool = True) -> pd.DataFrame:
    
    '''
    Reads CSV data for OMG data
    NB: data must be separated by " " separator

        Parameters:
                path_palm_data  (str): path to csv data file
                n_omg_channels  (int): Number of OMG channels
                n_acc_channels  (int): Number of Accelerometer channels, default = 0
                n_gyr_channels  (int): Number of Gyroscope channels, default = 0
                n_mag_channels  (int): Number of Magnetometer channels, default = 0
                n_enc_channels  (int): Number of Encoder channels, default = 0
                button_ch      (bool): If button channel is present, default = True
                sync_ch        (bool): If synchronization channel is present, default = True
                timestamp_ch   (bool): If timestamp channel is present, default = True

        Returns:
                df_raw (pd.DataFrame): Parsed pandas Dataframe with OMG data
    '''
    
    df_raw = pd.read_csv(path_palm_data, sep=' ', 
                         header=None, 
                         skipfooter=1, 
                         skiprows=1, 
                         engine='python')
    columns = np.arange(n_omg_channels).astype('str').tolist()
    
    for label, label_count in zip(['ACC', 'GYR', 'MAG', 'ENC'], 
                                  [n_acc_channels, n_gyr_channels, n_mag_channels, n_enc_channels]):
        columns = columns + ['{}{}'.format(label, i) for i in range(label_count)]
        
    if button_ch:
        columns = columns + ['BUTTON']
        
    if sync_ch:
        columns = columns + ['SYNC']
        
    if timestamp_ch:
        columns = columns + ['ts']
        
    df_raw.columns = columns
    
    return df_raw

#------------------------------------------------------------------------------------------------


def get_encoded_labels(gestures_protocol):
    le = LabelEncoder()

    # FIT
    le.fit(
        gestures_protocol[[
            "Thumb","Index","Middle","Ring","Pinky",
            'Thumb_stretch','Index_stretch','Middle_stretch','Ring_stretch','Pinky_stretch'
        ]]
        .apply(lambda row: str(tuple(row)), axis=1)
    )

    # TRANSFORM
    gestures_protocol['gesture'] = le.transform(
        gestures_protocol[[
            "Thumb","Index","Middle","Ring","Pinky",
            'Thumb_stretch','Index_stretch','Middle_stretch','Ring_stretch','Pinky_stretch'
        ]]
        .apply(lambda row: str(tuple(row)), axis=1)
    )
    
    return gestures_protocol

#------------------------------------------------------------------------------------------------


def count_adjacent_classes(vector):
    adj_classes = {}
    current_class = vector[0]
    count = 1
    class_count = {}

    for i in range(1, len(vector)):
        if vector[i] == current_class:
            count += 1
        else:
            key = f"{current_class}({class_count.get(current_class, 0)})"
            if count > 1:
                key = f"{current_class}({class_count.get(current_class, 0)})"
            adj_classes[key] = count
            class_count[current_class] = class_count.get(current_class, 0) + 1
            current_class = vector[i]
            count = 1

    key = f"{current_class}({class_count.get(current_class, 0)})"
    if count > 1:
        key = f"{current_class}({class_count.get(current_class, 0)})"
    adj_classes[key] = count
    class_count[current_class] = class_count.get(current_class, 0) + 1

    return adj_classes

#------------------------------------------------------------------------------------------------


GESTURES = ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']

def get_graphs(X_array, y_array, gestures_names=GESTURES, start=1000, end=1500):
    
    _, ax1 = plt.subplots(sharex=True, figsize=(12, 4))
    plt.suptitle(f'OMG and Protocol {start}:{end}')
    ax1.plot(X_array[start:end])
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('OMG')
    plt.grid(axis='x')
    ax2 = ax1.twinx()
    ax2.plot(y_array[start:end], 'b-')
    ax2.set_ylabel('y_cmd')
    plt.yticks(np.arange(len(gestures_names)), gestures_names)
    plt.show()
    
#------------------------------------------------------------------------------------------------
    
    
def get_naive_centering(
    X_arr, y_arr, gap=500, inter=1000,
    window=20, use_m=True, model=svm.SVC(), 
    return_metrics=False):
    """Функция для устранения глобального лага между сигналами датчиков и таргетом.

    Args:
        X_arr (ndarray): Массив данных.
        y_arr (ndarray): Вектор целевого признака.
        gap (int, optional): Размеры концевых отступов. Defaults to 500.
        inter (int, optional): Величина концевых выборок. Defaults to 1000.
        window (int, optional): Величина окна поиска оптимального сдвига. Defaults to 20.
        use_m (bool, optional): Использование модели для поиска оптимального сдвига.
            Defaults to True. False: поиск сдвига по корреляции таргета с вектором
            суммы модулей дифференциалов векторов признаков массива данных.
        model (_type_, optional): Алгоритм scikit-learn. Defaults to svm.SVC().
        return_metrics (bool, optional): Взвращение значений метрик

    Returns:
        tuple():
            ndarray: Вектор сдвинутого таргета.
            float: метрика на начальном участке.
            float: метрика на конечном участке.
        tuple():
            ndarray: Вектор сдвинутого таргета.
            list: Строки отчета по проделанным операциям.
    """
    # part of the data from the beginning
    X_part1 = X_arr[gap:gap+inter]
    y_part1 = y_arr[gap:gap+inter]
    # part of the data from the end
    X_part2 = X_arr[-gap-inter:-gap]
    y_part2 = y_arr[-gap-inter:-gap]
    
    # Функция для сдвига таргета
    def shifter(y_arr, shift=1):
        first_element = y_arr[0]
        prefix = np.full(shift, first_element)
        y_arr_shifted = np.concatenate((prefix, y_arr))[:-shift]
    
        return y_arr_shifted
    
    # Функция для расчета точности модели
    def get_score(X, y, model=model):
        model = model
        model.fit(X, y)
        preds = model.predict(X)
        
        return metrics.accuracy_score(y, preds)
    
    # Функция для расчета корреляции
    def get_corr(X, y):
        x_diff = pd.DataFrame(X).diff().abs().sum(axis=1)
        correlation = np.corrcoef(x_diff, y)[0, 1]
        
        return abs(correlation)
    
    
    max_score1, current_score1 = 0, 0
    max_score2, current_score2 = 0, 0
    s1, s2 = 1, 1
    
    for i in range(1, window+1):
        y_a = shifter(y_part1, shift=i)
        y_b = shifter(y_part2, shift=i)
        
        if use_m:
            current_score1 = get_score(X_part1, y_a)
            current_score2 = get_score(X_part2, y_b)
        else:
            current_score1 = get_corr(X_part1, y_a)
            current_score2 = get_corr(X_part2, y_b)
        
        if current_score1 > max_score1:
            max_score1, current_score1 = current_score1, max_score1
            s1 = i
        
        if current_score2 > max_score2:
            max_score2, current_score2 = current_score2, max_score2
            s2 = i
    
    optimal_shift = round((s1+s2)/2)
    y_arr_shifted = shifter(y_arr, shift=optimal_shift)
    summary = [
        f'Оптимальные свдиги для концевых выборок:   {s1} и {s2}\n',
        f'Accuracy/correlation на концевых выборках: {max_score1}; {max_score2}\n',
        f'Размер оптимального сдвига (как среднего): {optimal_shift}'
    ]
    
    if return_metrics:
        return y_arr_shifted, max_score1, max_score2
    
    return y_arr_shifted, summary

#------------------------------------------------------------------------------------------------


def find_highly_correlated_features(data, threshold=0.9):
    """Функция корреляционного анализа

    Args:
        data (ndarray): Массив данных.
        threshold (float, optional): Порог корреляции. Defaults to 0.9.

    Returns:
        tuple:
            list: Спсиок пар номеров высоко скореллированных признаков.
            set: Множество оптимально-отобранных номеров признаков из
                высоко скореллированных пар.
    """
    # Строим корреляционную матрицу
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    # Находим индексы нижнего треугольника корреляционной матрицы
    lower_triangle_indices = np.tril_indices(corr_matrix.shape[0], k=-1)
    
    # Находим пары высоко скоррелированных признаков
    high_corr_features = [
        [j, i] for i, j in zip(
            lower_triangle_indices[0], lower_triangle_indices[1]
        ) if abs(corr_matrix[i, j]) > threshold
    ]

    hcf_copy = high_corr_features.copy()
    # Определяем features_to_remove
    features_to_remove = []
    while True:
        dct = {}
        for el in hcf_copy:
            for num in el:
                cnt = 0
                for l in hcf_copy:
                    if (num in l) and (len(l) == 2):
                        cnt += 1
                dct[num] = cnt
        
        if len(set(dct.values())) == 1 and list(dct.values())[0] == 0:
            break

        if not dct:
            break
        
        num_tb_dltd = max(dct, key=dct.get)
        features_to_remove.append(num_tb_dltd)

        hcf_copy = [
            [x for x in inner_list if x != num_tb_dltd]
            for inner_list in hcf_copy
        ]

    return high_corr_features, list(set(features_to_remove))

#------------------------------------------------------------------------------------------------


def get_smoothing(Xdf, sample_size=5, alpha=0.3):
    """Функция для экспоненциального сглаживания

    Args:
        Xdf (ndarray): Массив данных.
        sample_size (int, optional): Размер сглаживаемой выборки. Defaults to 5.
        alpha (float, optional): Коэффициент сглаживания. Defaults to 0.3.

    Returns:
        ndarray: Преобразованный массив.
    """
    samples, j = [], 0
    for i in range(sample_size, Xdf.shape[0] + 1, sample_size):
        x_smoothed = Xdf.loc[j:i-1, list(Xdf.columns)].ewm(alpha=alpha, adjust=True).mean().values
        samples.append(x_smoothed)
        j = i
    
    X_ewm = np.row_stack(samples)
    
    return X_ewm

#------------------------------------------------------------------------------------------------


def get_convolve(data, M=10, tau=3, sym=True):
    win = signal.windows.exponential(M, tau=tau, sym=sym)
    data_result = np.zeros((data.shape[0]))
    for i in range(data.shape[1]):
        sig = data[:, i]
        filtered = signal.convolve(sig, win, mode='same') / sum(win)
        data_result = np.column_stack((data_result, filtered))
        
    return data_result[:, 1:]

#------------------------------------------------------------------------------------------------


def get_correlation(X, y):
    x_diff = pd.DataFrame(X).diff().abs().sum(axis=1)
    correlation = np.corrcoef(x_diff, y)[0, 1]
    
    return correlation

#------------------------------------------------------------------------------------------------


def get_runs_report(meta_data):
    
    report = []

    for i in range(len(meta_data)):
        
        file_name = meta_data['montage'][i]
        palm_file = './data/' + file_name
        
        file_name_for_report = file_name.split('.')[0]

        gestures = read_omg_csv(palm_file)

        gestures_protocol = pd.read_csv(f'{palm_file}.protocol.csv', index_col=0)
        gestures_protocol = get_encoded_labels(gestures_protocol)
        y = np.array([gestures_protocol['gesture'].loc[s] for s in gestures['SYNC'].values])
        X = gestures[[str(i) for i in range(50)]].values

        y_shifted, a_metric, b_metric = get_naive_centering(
            X, y, use_m=False, return_metrics=True
        )
        
        corr = get_correlation(X, y_shifted)

        last_train_idx = meta_data['last_train_idx'][i]

        X_train = X[:last_train_idx]
        X_test = X[last_train_idx:]

        y_train = y_shifted[:last_train_idx]
        y_test = y_shifted[last_train_idx:]

        _, features_to_remove = find_highly_correlated_features(X_train)

        X_train_cd = np.delete(X_train, features_to_remove, axis=1)
        X_test_cd = np.delete(X_test, features_to_remove, axis=1)

        std_scaler = StandardScaler()
        X_train_scaled = std_scaler.fit_transform(X_train_cd)
        X_test_scaled = std_scaler.transform(X_test_cd)

        pca = PCA(n_components=0.9999, random_state=42).fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        X_train_ce = get_convolve(X_train_pca, M=10, tau=1, sym=False)
        X_test_ce = get_convolve(X_test_pca, M=10, tau=1, sym=False)

        model = svm.SVC(
            kernel='rbf',
            random_state=42
        )

        # Обучение
        model.fit(X_train_ce, y_train)
        y_pred_test = model.predict(X_test_ce)

        f1_tests = []
        for j in range(6):
            f1_test = metrics.f1_score(y_test, y_pred_test, average=None)[j]
            f1_tests.append(f1_test)
            
        f1_test_weighted = metrics.f1_score(y_test, y_pred_test, average='weighted')

        new_row = {
            'file': file_name_for_report, 'Corr': corr, 'corr_a': a_metric,
            'corr_b': b_metric, 'f1_test': f1_test_weighted, 'Neutral': f1_tests[0],
            'Open': f1_tests[1], 'Pistol': f1_tests[2], 'Thumb': f1_tests[3],
            'OK': f1_tests[4], 'Grab': f1_tests[5]
        }
        
        report.append(new_row)

    report_df = pd.DataFrame(report)
    report_df = pd.concat([meta_data[['pilote_id']], report_df], axis=1)

    return report_df

#------------------------------------------------------------------------------------------------

def get_approx_lr_coefs(X, n_ftrs=10, prev=None, lin_alg=None):
    """Функция для формирования признакового описания 
       с использованием линейной регрессии

    Args:
        X (ndarray): массив данных.
        n_ftrs (int, optional): размер сэмпла. Defaults to 10.
        prev (ndarray, optional): предшествующие данные размером n_ftrs.
        Defaults to None.
        lin_alg (abc.ABCMeta, optional): линейный алгоритм sklearn.
        По умолчанию используется алгоритм МНК из numpy.

    Returns:
        ndarray: массив коэффициентов модели регрессии.
    """
    
    def get_coefs(data):
        # Разделение данных на признаки и целевую переменную
        segments_X = data[1:, :].T
        segments_y = data[0, :]
        
        # Обучение модели линейной регрессии
        if lin_alg is not None:
            model = lin_alg()
            model.fit(segments_X, segments_y)
            # Получение параметров модели
            return model.coef_
        else:
            w = np.linalg.lstsq(segments_X, segments_y, rcond=None)[0]
            # Вектор весов линейной функции
            return w
        
    
    w_g = np.zeros((0, n_ftrs))
    window = np.zeros((n_ftrs+1, X.shape[-1]))
    if prev is not None:
        window[1:, :] = prev
    
    for i in range(X.shape[0]):
        if not np.sum(window[0]):
            window[0, :] = X[i]
        else:
            window[1:, :] = window[:-1, :]
            window[0, :] = X[i]

        coefs = get_coefs(window)
        w_g = np.vstack((w_g, coefs))
    
    return w_g

#------------------------------------------------------------------------------------------------

def get_ar_coefs(array, p=5, prev=None):
    """Функция для генерации новых признаков на основе векторов
    оптимальных параметров модели авторегрессии.

    Args:
        array (ndarray): массив данных.
        p (int, optional): размер базы аппроксимации. Defaults to 5.
        prev (ndarray): массив данных для аппроксимации первой точки.
        Defaults to None.

    Returns:
        ndarray: массив сгенерированных признаков на основе
        рассчитанных коэффициентов.
    """
    new_features = np.zeros((array.shape[0],0))
    
    # Проходим внешним циклом по переменным (столбцам) в данных
    for i in range(array.shape[1]):
        if prev is None:
            data = np.hstack((np.zeros(p), array[:, i]))
        else:
            data = np.hstack((prev[:, i], array[:, i]))
        t = len(data)
        w_massive = np.zeros((0, p+1))
        
        # Проходим внутренним циклом по вектору значений переменного
        for j in range(p, t):
            features = data[j-p:j]
            labels = np.array(data[j]).reshape(-1, 1)
        
            # Решение системы уравнений методом наименьших квадратов
            X = np.hstack((np.ones(1), features)).reshape(1, -1)
            w = np.linalg.lstsq(X, labels, rcond=None)[0]
            
            w_massive = np.vstack((w_massive, w.reshape(1, -1)))
            
        # Собираем в единый массив параметры МНК
        new_features = np.hstack((new_features, w_massive))
        
    return new_features

#------------------------------------------------------------------------------------------------

def get_runs_report2(meta_data):
    """Функция для получения отчета по запускам (2 ред.)

    Args:
        meta_data (pd.DataFrame): таблица с данными о data-коллекциях.

    Returns:
        pd.DataFrame: отчет по запускам
    """
    
    report = []

    for i in tqdm(range(len(meta_data)), desc="Datasets processed"):
        
        file_name = meta_data['montage'][i]
        palm_file = './data/data_from_4_pilots/' + file_name
        
        file_name_for_report = file_name.split('.')[0]

        gestures = read_omg_csv(palm_file)

        gestures_protocol = pd.read_csv(f'{palm_file}.protocol.csv', index_col=0)
        gestures_protocol = get_encoded_labels(gestures_protocol)
        y = np.array([gestures_protocol['gesture'].loc[s] for s in gestures['SYNC'].values])
        X = gestures[[str(i) for i in range(50)]].values

        y_shifted, a_metric, b_metric = get_naive_centering(
            X, y, use_m=False, return_metrics=True
        )
        
        corr = get_correlation(X, y_shifted)

        last_train_idx = meta_data['last_train_idx'][i]

        X_train = X[:last_train_idx]
        X_test = X[last_train_idx:]

        y_train = y_shifted[:last_train_idx]
        y_test = y_shifted[last_train_idx:]

        mm_scaler = MinMaxScaler()
        X_train_mm = mm_scaler.fit_transform(X_train)
        X_test_mm = mm_scaler.transform(X_test)
        
        X_train_mm_ax = get_approx_lr_coefs(X_train_mm)
        X_test_mm_ax = get_approx_lr_coefs(X_test_mm)
        
        X_train_mm_ar = get_ar_coefs(X_train_mm)
        X_test_mm_ar = get_ar_coefs(X_test_mm)
        
        X_train_mm_ax_ar_ = np.hstack((X_train_mm, X_train_mm_ax, X_train_mm_ar))
        X_test_mm_ax_ar_ = np.hstack((X_test_mm, X_test_mm_ax, X_test_mm_ar))
        
        model = svm.SVC(
            kernel='rbf',
            random_state=42
        )

        # Обучение
        model.fit(X_train_mm_ax_ar_, y_train)
        y_pred_test = model.predict(X_test_mm_ax_ar_)

        f1_tests = []
        for j in range(6):
            f1_test = metrics.f1_score(y_test, y_pred_test, average=None)[j]
            f1_tests.append(f1_test)
            
        f1_test_weighted = metrics.f1_score(y_test, y_pred_test, average='weighted')

        new_row = {
            'file': file_name_for_report, 'Corr': corr, 'corr_a': a_metric,
            'corr_b': b_metric, 'f1_test': f1_test_weighted, 'Neutral': f1_tests[0],
            'Open': f1_tests[1], 'Pistol': f1_tests[2], 'Thumb': f1_tests[3],
            'OK': f1_tests[4], 'Grab': f1_tests[5]
        }
        
        report.append(new_row)

    report_df = pd.DataFrame(report)
    report_df = pd.concat([meta_data[['pilote_id']], report_df], axis=1)
    
    # Сохранение датафрейма в CSV файл
    report_df.to_csv('runs_report.csv', index=False)

    return report_df

#------------------------------------------------------------------------------------------------

models = [
    # Наивный байесовксий классификатор
    naive_bayes.GaussianNB(),
    
    # Линейная регрессия
    linear_model.LogisticRegression(
        solver='liblinear',
        max_iter=10000,
        random_state=42
    ),

    # Пассивно-агрессивный классификатор
    linear_model.PassiveAggressiveClassifier(random_state=42),

    # Простой перцептрон
    linear_model.Perceptron(random_state=42),

    # Линейный классификатор на Тихоновской регуляризации
    linear_model.RidgeClassifier(random_state=42),

    # Линейный классификатор на стохастическом градиентном спуске
    linear_model.SGDClassifier(random_state=42),

    # Линейный классификатор на опорных векторах
    svm.LinearSVC(
        dual='auto',
        random_state=42,
        max_iter=2000
    ),

    # Метод опорных векторов
    svm.SVC(random_state=42),

    # Классификатор на k-ближайших соседях
    neighbors.KNeighborsClassifier(),

    # Многослойный перцептрон
    neural_network.MLPClassifier(
        max_iter=2000,
        random_state=42
    ),

    # Дерево решений
    tree.DecisionTreeClassifier(random_state=42),

    # Классификатор адаптивного бустинга на деревьях решений
    ensemble.AdaBoostClassifier(
        estimator=tree.DecisionTreeClassifier(),
        algorithm='SAMME',
        random_state=42
    ),

    # Простой бэггинг на деревьях решений
    ensemble.BaggingClassifier(
        estimator=tree.DecisionTreeClassifier(),
        n_jobs=-1,
        random_state=42
    ),

    # Классификатор чрезвычайно рандомизированных деревьев
    ensemble.ExtraTreesClassifier(
        n_jobs=-1,
        random_state=42
    ),

    # Градиентный бустинг
    ensemble.GradientBoostingClassifier(random_state=42),

    # Лес случайных деревьев
    ensemble.RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    ),

    # Градиентный бустинг на гистограммах
    ensemble.HistGradientBoostingClassifier(random_state=42),

    # Крутой градиентный бустинг :)
    xgb.XGBClassifier(seed=42, n_jobs=-1),

    # Градиентный бустинг от Microsoft
    lgbm.LGBMClassifier(
        objective='multiclass',
        seed=42
    ),

    # Градиентный бустинг от Яндекс
    cb.CatBoostClassifier(verbose=False, random_seed=42)
]



def get_runs_report3(meta_data, models=models):
    """Функция для получения отчета по запускам (3 ред.)

    Args:
        meta_data (pd.DataFrame): таблица с данными о data-коллекциях.
        models (list, optional): список моделей. Defaults to models.

    Returns:
        touple:
            ndarray: массив отчета;
            list:    список моделей
    """

    reports = []
    
    for model in tqdm(models, desc="Models processed"):
    
        lst = []

        for i in range(len(meta_data)):
            
            file_name = meta_data['montage'][i]
            palm_file = './data/data_from_4_pilots/' + file_name

            gestures = read_omg_csv(palm_file)

            gestures_protocol = pd.read_csv(f'{palm_file}.protocol.csv', index_col=0)
            gestures_protocol = get_encoded_labels(gestures_protocol)
            y = np.array([gestures_protocol['gesture'].loc[s] for s in gestures['SYNC'].values])
            X = gestures[[str(i) for i in range(50)]].values

            y_shifted, a_metric, b_metric = get_naive_centering(
                X, y, use_m=False, return_metrics=True
            )
            
            corr = get_correlation(X, y_shifted)

            last_train_idx = meta_data['last_train_idx'][i]

            X_train = X[:last_train_idx]
            X_test = X[last_train_idx:]

            y_train = y_shifted[:last_train_idx]
            y_test = y_shifted[last_train_idx:]

            mm_scaler = MinMaxScaler()
            X_train_mm = mm_scaler.fit_transform(X_train)
            X_test_mm = mm_scaler.transform(X_test)
            
            X_train_mm_ax = get_approx_lr_coefs(X_train_mm)
            X_test_mm_ax = get_approx_lr_coefs(X_test_mm)
            
            X_train_mm_ar = get_ar_coefs(X_train_mm)
            X_test_mm_ar = get_ar_coefs(X_test_mm)
            
            X_train_mm_ax_ar_ = np.hstack((X_train_mm, X_train_mm_ax, X_train_mm_ar))
            X_test_mm_ax_ar_ = np.hstack((X_test_mm, X_test_mm_ax, X_test_mm_ar))

            # Обучение
            model.fit(X_train_mm_ax_ar_, y_train)
            y_pred_test = model.predict(X_test_mm_ax_ar_)

            f1_tests = []
            for j in range(6):
                f1_test = metrics.f1_score(y_test, y_pred_test, average=None)[j]
                f1_tests.append(f1_test)
                
            f1_test_weighted = metrics.f1_score(y_test, y_pred_test, average='weighted')
            
            data_row = np.array([
                corr, a_metric, b_metric,
                f1_test_weighted, 
                f1_tests[0], f1_tests[1], f1_tests[2], 
                f1_tests[3], f1_tests[4], f1_tests[5]
            ])
            
            lst.append(data_row)
            
        reports.append(np.array(lst))
        
    reports_array = np.array(reports)
    np.save('reports.npy', reports_array)

    return reports_array, models

#------------------------------------------------------------------------------------------------

def get_target(meta_data):
    """Функция для получения единого вектора 
    целовой перемнной по всем наборам данных.

    Args:
        meta_data (pd.DataFrame): таблица с данными о data-коллекциях.

    Returns:
        ndarray: вектор целевой переменной со всеми метками классов.
    """
    
    target = np.zeros(0)
    
    for i in range(len(meta_data)):
            
            file_name = meta_data['montage'][i]
            palm_file = './data/data_from_4_pilots/' + file_name

            gestures = read_omg_csv(palm_file)

            gestures_protocol = pd.read_csv(f'{palm_file}.protocol.csv', index_col=0)
            gestures_protocol = get_encoded_labels(gestures_protocol)
            y = np.array([gestures_protocol['gesture'].loc[s] for s in gestures['SYNC'].values])
            X = gestures[[str(i) for i in range(50)]].values

            y_shifted, _ = get_naive_centering(X, y, use_m=False)
            
            target = np.concatenate((target, y_shifted), axis=None)
    
    return target

#------------------------------------------------------------------------------------------------

def get_data_homogeneity_report(meta_data):
    report = []

    for i in range(len(meta_data)):
        
        file_name = meta_data['montage'][i]
        # Путь к файлу
        data_path = './data/data_from_4_pilots/' + file_name
        file_name_for_report = file_name.split('.')[0]
        
        # Загрузка данных
        data = pd.read_csv(data_path, sep=' ').to_numpy()[:, :50]
        
        # Индекс отсечения
        last_train_idx = meta_data['last_train_idx'][i]
        
        # Формирование вектора лейблов принадлежности к выборке
        target = np.array([0 if i <= last_train_idx else 1 for i in range(len(data))])
        
        # Воссоединение данных с таргетом для перемешивания
        data_with_target = np.concatenate((data, target.reshape(-1, 1)), axis=1)
        
        # Перемешивание
        np.random.shuffle(data_with_target)
        
        # Разделение данных и таргета
        shuffled_data = data_with_target[:, :-1]
        shuffled_target = data_with_target[:, -1]
        
        # Сэмплирование
        X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_target, test_size=0.2, random_state=42)

        # Создание и обучение модели
        model = linear_model.LogisticRegression(solver='liblinear', max_iter=500)
        model.fit(X_train, y_train)

        # Предсказание принадлежности строк данных к выборкам
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Расчет и сохранение метрик
        f1_train_class_0 = metrics.f1_score(y_train, train_preds, pos_label=0)
        f1_train_class_1 = metrics.f1_score(y_train, train_preds, pos_label=1)
        f1_test_class_0 = metrics.f1_score(y_test, test_preds, pos_label=0)
        f1_test_class_1 = metrics.f1_score(y_test, test_preds, pos_label=1)
        
        new_row = {
                'file': file_name_for_report,
                'f1 train class (train)': f1_train_class_0,
                'f1 test class (train)': f1_train_class_1,
                'f1 train class (test)': f1_test_class_0,
                'f1 test class (test)': f1_test_class_1
            }
        
        report.append(new_row)
        
    report_df = pd.DataFrame(report)
    report_df = pd.concat([meta_data[['pilote_id']], report_df], axis=1)

    return report_df

