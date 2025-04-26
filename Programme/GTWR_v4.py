import os
import sqlite3
import time
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
from numba import njit, prange
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import brute
from skopt.space import Real, Integer
from skopt import gp_minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


def normalize_data(input_data, params_data):
    """
   Normalize input data based on specified normalization parameters for each feature.

   Parameters:
   ----------
   input_data : pandas.DataFrame
   params_data : list
       A list of parameter objects need to be normalized.

   Returns:
   -------
   pandas.DataFrame
       The input dataframe with normalized values for specified columns.
   """
    for param in params_data:
        input_data[param.name] = (input_data[param.name] - param.normal_params[0]) / (param.normal_params[1] - param.normal_params[0])

    return input_data


def denormalize_data(input_data, params_data):
    for param in params_data:
        original_min, original_max = param.normal_params[0]
        original_abs = max(abs(original_min), abs(original_max))
        if original_min * original_max < 0:
            original_min, original_max = -original_abs, original_abs
        target_min, target_max = param.normal_params[1]

        # Reverse standardization formula
        input_data[param.name] = ((input_data[param.name] - target_min) / (target_max - target_min)) * (
                original_max - original_min) + original_min

    return input_data


def random_cross_validation_split(data_df, n_splits=10, random_state=42):
    """
    Randomly shuffle and split the dataset into train and test sets for cross-validation.

    Parameters:
    ----------
    data_df : pandas.DataFrame
        The input dataset as a pandas DataFrame, containing features and target columns.
    n_splits : int, optional
        The number of splits (folds) for cross-validation, default is 10.
    random_state : int, optional
        Random seed to ensure reproducibility, default is 42.

    Returns:
    -------
    generator
        A generator yielding a tuple of `(train_data, test_data)` for each fold.
    """
    np.random.seed(random_state)
    data_shuffled = data_df.sample(frac=1).reset_index(drop=True)

    split_size = len(data_shuffled) // n_splits
    for i in range(n_splits):
        start_index = i * split_size
        if i == n_splits - 1:
            end_index = len(data_shuffled)
        else:
            end_index = start_index + split_size

        test_data = data_shuffled.iloc[start_index:end_index]
        train_data = pd.concat([data_shuffled.iloc[:start_index], data_shuffled.iloc[end_index:]])

        test_data = test_data.copy()
        test_data['cv_group'] = i + 1

        yield train_data, test_data


@njit
def st_distance(a_space, a_time, b_space, b_time, scale):
    """
   Calculates the spatiotemporal distance between two sets of points.

   Parameters:
   ----------
   a_space, a_time : numpy.ndarray
       2D array of spatial coordinates (x, y), time for set A.
   b_space, b_time : numpy.ndarray
       2D array of spatial coordinates (x, y), time for set B.
   scale : float
       Scaling factor for the spatial distances.

   Returns:
   -------
   st_dists : numpy.ndarray
       2D array of spatiotemporal distances.
   """
    n_a = a_space.shape[0]
    n_b = b_space.shape[0]

    # 1. 计算空间距离
    space_dists = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            dx = a_space[i, 0] - b_space[j, 0]
            dy = a_space[i, 1] - b_space[j, 1]
            space_dists[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    # 2. 计算时间距离
    time_dists = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            time_dists[i, j] = np.abs(a_time[i] - b_time[j])

    # 3. 归一化（可选）
    if NORMALIZE_SIGN:
        min_space = np.min(space_dists)
        max_space = np.max(space_dists)
        if max_space != min_space:
            space_dists = (space_dists - min_space) / (max_space - min_space)

        min_time = np.min(time_dists)
        max_time = np.max(time_dists)
        if max_time != min_time:
            time_dists = (time_dists - min_time) / (max_time - min_time)

    # 4. 合并距离
    return scale * space_dists + time_dists


# @njit(parallel=True)
def gtwr_chunk(predict_points, known_points, params, x_num):
    """
    Compute the Geographically and Temporally Weighted Regression (GTWR) predictions for a chunk of points.

    Parameters:
    ----------
    predict_points, known_points : numpy.ndarray
    params : tuple
        A tuple containing the scale parameter for spatial distance and the number of nearest neighbors (q).
    x_num : int
        The number of features used for prediction (excluding the intercept term).

    Returns:
    -------
    numpy.ndarray
        A 1D array containing the predicted values for each point in `predict_points`.
    """
    predict_time_cols, predict_cord_cols, predict_x_cols = predict_points
    known_time_cols, known_cord_cols, known_x_cols, known_y_cols = known_points
    scale, q = params
    n = predict_time_cols.shape[0]  # Number of prediction points

    # Compute the spatiotemporal distance matrix between prediction and known points
    distance_matrix = st_distance(predict_cord_cols, predict_time_cols,
                                  known_cord_cols, known_time_cols, scale)

    predictions = np.empty(n)

    for i in prange(n):  # Parallel loop for each prediction point
        distance = distance_matrix[i, :]
        distance[distance == 0] = np.inf  # Avoid division by zero for identical points
        non_infinite_count = (~np.isinf(distance)).sum()
        effective_q = min(q, non_infinite_count)
        if effective_q < q:
            predictions[i] = np.nan
            continue

        # Get indices of q nearest neighbors
        smallest_indices = np.argsort(distance)[:effective_q]
        max_ds = distance[smallest_indices[-1]]  # Maximum distance among nearest neighbors
        distance = distance[smallest_indices]

        # Compute weight matrix W based on the distances
        W = np.exp(-np.square(distance) / (max_ds ** 2))
        W_diag = np.diag(W)

        # Construct the design matrix X with intercept (first column = 1)
        X = np.empty((effective_q, x_num + 1))
        X[:, 0] = 1  # Intercept term
        X[:, 1:] = known_x_cols[smallest_indices]  # Use feature columns

        y = known_y_cols[smallest_indices]  # Target variable (assumed to be in the second-to-last column)

        # Perform weighted least squares regression
        XTWX = X.T @ W_diag @ X
        XTWy = X.T @ W_diag @ y
        beta = np.linalg.pinv(XTWX) @ XTWy  # Compute regression coefficients

        # Predict value for the current point
        X_with_intercept = np.concatenate((np.array([1]), predict_x_cols[i]))
        y_pred = X_with_intercept @ beta
        predictions[i] = y_pred

    return predictions


def gtwr(predict_points, known_points, params, predict_num, x_num, batch_size=1000):
    """
    Perform GTWR over a large set of points, processing in batches.

    Parameters:
    ----------
    predict_points, known_points : numpy.ndarray
    params : tuple
        A tuple containing the scale for spatial distances and the number of nearest neighbors (q).
    x_num : int
        The number of features used for prediction (excluding the intercept term).
    batch_size : int, optional
        The number of points to process in each batch (default is 1000).

    Returns:
    -------
    numpy.ndarray
        A 1D array of predicted values for all points in `predict_points`.
    """
    scale, q = params
    q = int(q)

    # Handle invalid parameters by returning a large error value
    if scale <= 0:
        return np.full(predict_num, 1e10)

    predictions = np.empty(predict_num)

    # Process points in batches to save memory
    for start in range(0, predict_num, batch_size):
        end = min(start + batch_size, predict_num)
        predict_points_chunk = tuple(arr[start:end] for arr in predict_points)
        chunk_predictions = gtwr_chunk(predict_points_chunk, known_points, (scale, q), x_num)
        predictions[start:end] = chunk_predictions

    return predictions


def extract_columns_to_numpy(points, t_col, coord_cols, x_cols, y_col=None):
    t_values = points[t_col].to_numpy()
    coords_values = points[coord_cols].to_numpy()
    x_values = points[x_cols].to_numpy()
    if y_col:
        y_values = points[y_col].to_numpy()
        return t_values, coords_values, x_values, y_values
    else:
        return t_values, coords_values, x_values


def gtwr_cv(points, cv, params, aux_list, x_num, y_name):
    """
    Perform cross-validation or fitting result for GTWR model.

    Parameters:
    ----------
    points : pandas.DataFrame
    cv : int
    params : tuple
    x_num : int
    y_name : str

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the GTWR predictions for each point, with cross-validation group labels if applicable.
    """
    start_time = time.time()
    all_results = pd.DataFrame()

    # Calculate for fitting result
    if cv == 1:
        print(f'\rCalculate fitting result for sample data.', end='', flush=True)
        predict_num = points.shape[0]
        test_data = extract_columns_to_numpy(points, 't', ['lon', 'lat'], aux_list)
        train_data = extract_columns_to_numpy(points, 't', ['lon', 'lat'], aux_list, y_name)
        test_result = gtwr(test_data, train_data, params, predict_num, x_num)

        # Create a copy of the group and store the GTWR predictions
        all_results = points.copy()
        all_results['gtwr'] = test_result

    # Calculate for verify result
    else:
        # Iterate through training and test splits generated by a random CV splitter
        st_scale, q = params
        for train_group, test_group in random_cross_validation_split(points, cv):
            print(f'\rParameters - st_scale:{st_scale:.4f}, q:{int(q):2d}', end='', flush=True)

            # Perform GTWR for the test set
            predict_num = test_group.shape[0]
            test_data = extract_columns_to_numpy(test_group, 't', ['lon', 'lat'], aux_list)
            train_data = extract_columns_to_numpy(train_group, 't', ['lon', 'lat'], aux_list, y_name)
            test_result = gtwr(test_data, train_data, params, predict_num, x_num)

            # Create a copy of the test group and store the GTWR predictions
            result_df = test_group.copy()
            result_df['gtwr'] = test_result
            result_df['cv_group'] = test_group['cv_group'].values

            # Concatenate the results of each fold to the final DataFrame
            all_results = pd.concat([all_results, result_df], ignore_index=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute the Root Mean Squared Error (RMSE) between actual and predicted values
    rmse = np.sqrt(mean_squared_error(all_results.dropna()[y_name], all_results.dropna()['gtwr']))
    print(f' >>>> time: {elapsed_time:.2f}s, RMSE: {rmse:.4f}')

    return all_results


def save_accurancy_pic(data1, data2, save_path, pics, models=None):
    real_name, pred_name, param_name = pics
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if models:
        fig.subplots_adjust(bottom=0.2)

    titles = ["Verification Results", "Fitting Results"]
    data_list = [data1, data2]

    for i, (ax, data, title) in enumerate(zip(axes, data_list, titles)):
        actual = data.dropna()[real_name].values
        predicted = data.dropna()[pred_name].values

        reg = LinearRegression()
        reg.fit(actual.reshape(-1, 1), predicted)

        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        sns.scatterplot(x=actual, y=predicted, color='blue', s=10, alpha=0.5, edgecolor=None, ax=ax)
        ax.plot([0, max(predicted)], [0, max(predicted)], 'k-', label='1:1 line')

        x_vals = np.array(ax.get_xlim())
        y_vals = reg.predict(x_vals.reshape(-1, 1))
        ax.plot(x_vals, y_vals, 'c--', label=f"Y = {reg.coef_[0]:.2f}X + {reg.intercept_:.2f}")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, min(max(actual), max(predicted)))
        ax.set_ylim(0, min(max(actual), max(predicted)))

        ax.text(0.05, 0.95, f'N = {len(actual)}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.90, f'R² = {r2:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.85, f'RMSE = {rmse:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.80, f'MAE = {mae:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.75, f'MAPE = {mape:.2f}%', fontsize=10, transform=ax.transAxes)

        ax.set_xlabel(f'Actual {param_name} (μg/m³)')
        ax.set_ylabel(f'Predicted {param_name} (μg/m³)')
        ax.set_title(title)
        ax.legend()

    if models:
        opt, aux_list = models
        st_scale, q_num = opt
        plt.figtext(0.5, 0.08, f'Optimal params -- scale: {st_scale:.4f}, q: {q_num}', ha="center", fontsize=10)
        plt.figtext(0.5, 0.05, 'Auxiliary variables-- %s' % ', '.join(aux_list), ha="center", fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


class Database:
    def __init__(self, db):
        self.db = db

    def create_table(self, table_name, fields):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            create_sql = f'''CREATE TABLE IF NOT EXISTS {table_name}({','.join([' '.join(field) for field in fields])})'''
            cursor.execute(create_sql)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def update_table(self, table_name, update_items):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            for update_item in update_items:
                alter_clause = ', '.join([f"{col} = ?" for col, _ in update_item['alter']])
                condition_clause = ' AND '.join([f"{col} = ?" for col, _ in update_item['condition']])
                update_sql = f"UPDATE {table_name} SET {alter_clause} WHERE {condition_clause}"
                params = [value for _, value in update_item['alter']] + [value for _, value in update_item['condition']]
                cursor.execute(update_sql, params)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def insert_table(self, table_name, insert_items):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            for insert_item in insert_items:
                columns = ', '.join([col for col, _ in insert_item])
                placeholders = ', '.join(['?' for _ in insert_item])
                values = [value for _, value in insert_item]
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def execute_sql(self, sql_sen):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_sen)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            conn.commit()
            return df
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def csv_to_sqlite(self, csv_file, param_names, table_name):
        df = pd.read_csv(csv_file)

        if param_names and len(param_names) == 4:
            id_col, t_col, x_col, y_col = param_names

            if x_col in df.columns and y_col in df.columns:
                rename_dict = {
                    id_col: 'id',
                    t_col: 't',
                    x_col: 'lon',
                    y_col: 'lat'
                }
                df.rename(columns=rename_dict, inplace=True)
            else:
                raise ValueError(f'The column name {param_names} is not in the CSV file')

        conn = sqlite3.connect(self.db)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        conn.close()


class AuxVar:
    def __init__(self, name, time_sign, time_gap, normal_params=None, nan_data=None):
        self.name = name
        self.time_sign = time_sign
        self.time_gap = time_gap
        self.normal_params = normal_params
        if nan_data:
            self.nan_data = nan_data

    def set_normal_params(self, database, table_name):
        sql = f'select max({self.name}) as max, min({self.name}) as min from {table_name}'
        max_min = database.execute_sql(sql)
        self.normal_params = [max_min['min'].iloc[0], max_min['max'].iloc[0]]


class GTWR:
    def __init__(self, name, base_dir, predict_item, aux_var_list):
        self.name = name
        self.base_dir = base_dir

        self.db = Database(os.path.join(base_dir, fr"{self.name}.db"))

        self.shp_dir = os.path.join(base_dir, "shp")
        self.shp_grid_point = os.path.join(self.shp_dir, "grid_points.shp")
        self.minx, self.miny, self.maxx, self.maxy = gpd.read_file(self.shp_grid_point).total_bounds

        self.raster_dir = os.path.join(base_dir, "raster")

        self.result_dir = os.path.join(base_dir, "result")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.accuracy_dir = os.path.join(base_dir, "accuracy")
        if not os.path.exists(self.accuracy_dir):
            os.makedirs(self.accuracy_dir)

        self.predict_item = predict_item
        self.aux_var_list = aux_var_list
        self.aux_var_size = len(self.aux_var_list)
        self.aux_var_names = [aux_var.name for aux_var in self.aux_var_list]

        self.st_var_list, self.known_points = None, None

    def set_spore(self, st_scale, q_num):
        dir_name = f"scale{st_scale:.2f}_q{q_num}"
        result_bath_dir = os.path.join(self.base_dir, "result")
        self.result_dir = os.path.join(result_bath_dir, dir_name)
        if not os.path.exists(result_bath_dir):
            os.makedirs(result_bath_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        accuracy_bath_dir = os.path.join(self.base_dir, "accuracy")
        self.accuracy_dir = os.path.join(accuracy_bath_dir, dir_name)
        if not os.path.exists(self.accuracy_dir):
            os.makedirs(self.accuracy_dir)

    def prepare_sample_data(self, from_csv=None, param_names=None, table_name='sample_data'):
        if from_csv:
            self.db.csv_to_sqlite(from_csv, param_names, table_name)
        # Create SQL sentence and execute
        x_array, y_array = [aux_var.name for aux_var in self.aux_var_list], [self.predict_item]
        sql = 'select id, t, lon, lat, {} from {}'.format(','.join(x_array + y_array), table_name)
        sql_point = self.db.execute_sql(sql)
        if CHECK_COLLINEAR:
            removes = self.remove_high_vif(sql_point)
            sql_point = sql_point.drop(columns=removes)
            self.aux_var_list = [aux_var for aux_var in self.aux_var_list if aux_var.name not in removes]
            self.aux_var_names = [aux_var.name for aux_var in self.aux_var_list]
            self.aux_var_size = len(self.aux_var_list)

        # Handling null values
        sql_point = sql_point.dropna()
        for param in self.aux_var_list:
            if getattr(param, 'nan_data', None):
                sql_point = sql_point[~sql_point[[param.name]].isin([param.nan_data]).any(axis=1)]

        if NORMALIZE_SIGN == 'all':
            for aux_var in self.aux_var_list:
                if aux_var.normal_params is None:
                    aux_var.set_normal_params(self.db, table_name)
            return normalize_data(sql_point, self.aux_var_list)
        else:
            return sql_point

    def remove_high_vif(self, df, threshold=10):
        def calculate_vif(df):
            vif_data = pd.DataFrame()
            vif_data['Feature'] = df.columns
            vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
            return vif_data

        df = df[self.aux_var_names]
        df = df.dropna()
        vif_data = calculate_vif(df)
        removed_fefatures = []
        while vif_data['VIF'].max() > threshold:
            max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            removed_fefatures.append(max_vif_feature)
            # print(f"Removing {max_vif_feature} with VIF: {vif_data['VIF'].max()}")

            df = df.drop(columns=[max_vif_feature])

            vif_data = calculate_vif(df)
        print('---------------------Collinearity check---------------------')
        print(vif_data)
        print(f'Remove {removed_fefatures}')
        return removed_fefatures

    # RMSE
    def objective_function_RMSE(self, params):
        predict = gtwr_cv(self.known_points, 10, params, self.aux_var_names, self.aux_var_size, self.predict_item)
        return mean_squared_error(predict.dropna()[self.predict_item], predict.dropna()['gtwr'])

    def save_result_each(self, params):
        st_scale, q_num = params
        self.set_spore(st_scale, q_num)     # Prepare store dir
        # Calculate verify matters for GTWR result
        print('Calculate verification result for sample data.\t', end='')
        verify_result = gtwr_cv(self.known_points, 10, (st_scale, q_num),
                                self.aux_var_names, self.aux_var_size, self.predict_item)
        verify_file_path = os.path.join(self.accuracy_dir, 'result_gtwr_verify.csv')
        with open(verify_file_path, 'w', encoding='utf-8') as f:
            f.write(f"scale: {st_scale}, q: {q_num}\n")
        verify_result.to_csv(verify_file_path, mode='a', index=False, encoding='utf-8')

        # Calculate fitting matters for GTWR result
        fitting_result = gtwr_cv(self.known_points, 1, (st_scale, q_num),
                                 self.aux_var_names, self.aux_var_size, self.predict_item)
        fitting_file_path = os.path.join(self.accuracy_dir, 'result_gtwr_fitting.csv')
        with open(fitting_file_path, 'w', encoding='utf-8') as f:
            f.write(f"scale: {st_scale}, q: {q_num}\n")
        fitting_result.to_csv(fitting_file_path, mode='a', index=False, encoding='utf-8')

        # Save accuracy picture
        accuracy_pic_path = os.path.join(self.accuracy_dir,
                                         f"scale{st_scale:.2f}_q{q_num}.png")
        pic_params = (self.predict_item, 'gtwr', self.predict_item)
        model_params = ((st_scale, q_num), self.aux_var_names)
        save_accurancy_pic(verify_result, fitting_result, accuracy_pic_path, pic_params, model_params)

    def extract_data_from_grid(self, group, time):
        grid_points = gpd.read_file(self.shp_grid_point).drop(columns=['geometry'])
        # year, day = group['year'].values[0], group['day'].values[0]
        grid_points.insert(1, 't', time)
        # grid_points.insert(2, 'day', day)

        for param in self.aux_var_list:
            name = param.name
            time_sign, time_gap = param.time_sign, param.time_gap
            raster_dir = os.path.join(self.raster_dir, name)
            # Need to adjust according to the actual time interval of the grid
            file_time_sign = f'y{time//356+1}' if time_sign == 'year' else f't{time-(time%time_gap)+time_gap//16}'

            raster_file = os.path.join(raster_dir, f"{name}_{file_time_sign}.tif")
            if os.path.exists(raster_file):
                with rasterio.open(raster_file) as raster:
                    lon, lat = grid_points['lon'].values, grid_points['lat'].values
                    rows, cols = raster.index(lon, lat)
                    raster_data = raster.read(1)
                    raster_values = raster_data[rows, cols]
                    grid_points[param.name] = raster_values

        desired_order = ['id', 't', 'lon', 'lat'] + self.aux_var_names
        grid_points = grid_points[desired_order]
        grid_points['t'] = time

        # Handling null values
        grid_points = grid_points.dropna()
        for param in self.aux_var_list:
            if getattr(param, 'nan_data', None):
                grid_points = grid_points[~grid_points[[param.name]].isin([param.nan_data]).any(axis=1)]

        if NORMALIZE_SIGN is None:
            return grid_points
        elif NORMALIZE_SIGN == 'all':
            return normalize_data(grid_points, self.aux_var_list)

    def gtwr_grid(self, params, time_range=None, table_name='sample_data'):
        predict_points_time = pd.DataFrame()
        predict_points_time['t'] = time_range

        grouped = predict_points_time.groupby('t')
        for time, group in list(grouped):
            t = group['t'].values[0]
            print(f'\rCalculate GTWR grid for {t} -- Extract Aux from raster', end='', flush=True)
            predict_points = self.extract_data_from_grid(group, t)

            print(f'\rCalculate GTWR grid for {t} -- Calculate grid value', end='', flush=True)
            predict_num = predict_points.shape[0]
            test_data = extract_columns_to_numpy(predict_points, 't', ['lon', 'lat'], self.aux_var_names)
            train_data = extract_columns_to_numpy(self.known_points, 't', ['lon', 'lat'], self.aux_var_names, self.predict_item)
            predictions = gtwr(test_data, train_data, params, predict_num, self.aux_var_size)

            coords = predict_points[['lon', 'lat']].values

            geometries = [Point(xy) for xy in coords]
            gdf = gpd.GeoDataFrame({'lon': coords[:, 0], 'lat': coords[:, 1], 'gtwr': predictions.flatten()},
                                   geometry=geometries)
            gdf.set_crs(epsg=4326, inplace=True)

            output_file = os.path.join(self.result_dir, f"gtwr_t{t}.shp")
            gdf.to_file(output_file)


if __name__ == '__main__':
    ''' Static parameters '''
    # Predict time range [year, day of year]
    PREDICT_TIME_RANGE = [357, 358]

    # NORMALIZE_SIGN determines the method of normalization:
    # None  - No normalization applied.
    # 'st'  - Only normalize spatial and temporal items.
    # 'all'  - Normalize independent variable and spatiotemporal items.
    NORMALIZE_SIGN = 'all'

    # Eliminate variables with high collinearity
    CHECK_COLLINEAR = True

    ''' Parameters of auxiliary variables '''
    tno2 = AuxVar(name='tno2', time_sign='day', time_gap=1)
    temp = AuxVar(name='temp', time_sign='day', time_gap=1)
    et = AuxVar(name='et', time_sign='day', time_gap=1)
    sp = AuxVar(name='sp', time_sign='day', time_gap=1)
    tp = AuxVar(name='tp', time_sign='day', time_gap=1)
    ws = AuxVar(name='ws', time_sign='day', time_gap=1)
    pop = AuxVar(name='pop', time_sign='year', time_gap=1)
    building = AuxVar(name='building', time_sign='year', time_gap=1)
    road = AuxVar(name='road', time_sign='year', time_gap=1)
    ndvi = AuxVar(name='ndvi', time_sign='day', time_gap=16, nan_data=-3000)

    ''' Create GTWR object '''
    gtwr_obj = GTWR(name='example',
                    base_dir=r"example_data",
                    predict_item='NO2',
                    aux_var_list=[tno2, temp, et, sp, tp, ws, pop, building, road, ndvi])

    ''' Prepare the sample data'''
    gtwr_obj.known_points = gtwr_obj.prepare_sample_data(from_csv=r"example_data/sample_data_csv.csv",
                                                         param_names=['station_id', 'time_sign', 'x', 'y'],
                                                         table_name='sample_data_csv')

    ''' Calculate the best params for GTWR model '''
    print('\n-------------Finding the Best Optimal Parameters-------------')
    search_space = [Real(0.00001, 1, name='st_scale'), Integer(10, 100, name='q_num')]
    result = gp_minimize(gtwr_obj.objective_function_RMSE, search_space, n_calls=200, random_state=0)
    initial_guess = result.x
    optimal_params = fmin(gtwr_obj.objective_function_RMSE, initial_guess, disp=True)

    print(f"best st_scale: {optimal_params[0]}, best q_num: {int(optimal_params[1])}")
    st_scale_optimal, q_num_optimal = optimal_params[0], int(optimal_params[1])

    ''' Save verification and fitting result for each time '''
    print('\n----------------Finding the Optimal Parameters----------------')
    gtwr_obj.save_result_each((st_scale_optimal, q_num_optimal))

    ''' Calculate Grid GTWR result'''
    print('\n--------------------Calculate grids result--------------------')
    gtwr_obj.gtwr_grid((st_scale_optimal, q_num_optimal), PREDICT_TIME_RANGE, table_name='sample_data_csv')
    print('\n-----------------------------DONE-----------------------------')
