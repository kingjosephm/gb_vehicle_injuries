import pandas as pd
from typing import Dict, List
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)


def read_data() -> pd.DataFrame:
    """
    Reads 3 years of data from disk (requires downloading manually ahead of time to ./data) and preprocesses.
    :return: pd.DataFrame
    """
    df = pd.DataFrame()
    for i in range(2019, 2022):

        # Read data & drop select columns
        vehicle = pd.read_csv(f"./data/dft-road-casualty-statistics-vehicle-{i}.csv", low_memory=False)
        vehicle = drop_columns(vehicle, 'vehicle')
        accident = pd.read_csv(f"./data/dft-road-casualty-statistics-accident-{i}.csv", low_memory=False)
        accident = drop_columns(accident, 'accident')
        casualty = pd.read_csv(f"./data/dft-road-casualty-statistics-casualty-{i}.csv", low_memory=False)
        casualty = drop_columns(casualty, 'casualty')

        #######################
        ###### Data Mods  #####
        #######################

        # Recode vehicle type
        vehicle['vehicle_type'] = vehicle['vehicle_type'].replace(recode_vehicle_type())

        # Recode unknowns to missing
        # Note - I initially coded missings as -1, but LGBM handles np.NaN natively, so I convert them to this below
        for col in ['towing_and_articulation', 'junction_location', 'skidding_and_overturning',
                    'vehicle_leaving_carriageway', 'first_point_of_impact', 'vehicle_left_hand_drive']:
            vehicle[col] = np.where(vehicle[col] == 9, -1, vehicle[col])
        for col in ['vehicle_manoeuvre', 'vehicle_location_restricted_lane', 'hit_object_in_carriageway',
                    'hit_object_off_carriageway']:
            vehicle[col] = np.where(vehicle[col] == 99, -1, vehicle[col])
        for col in ['journey_purpose_of_driver']:
            vehicle[col] = np.where(vehicle[col] == 6, -1, vehicle[col])
        for col in ['sex_of_driver']:
            vehicle[col] = np.where(vehicle[col] == 3, -1, vehicle[col])
        for col in ['road_type', 'junction_control', 'weather_conditions', 'road_surface_conditions',
                    'special_conditions_at_site', 'carriageway_hazards']:
            accident[col] = np.where(accident[col] == 9, -1, accident[col])
        for col in ['speed_limit']:
            accident[col] = np.where(accident[col] == 99, -1, accident[col])
        vehicle['journey_purpose_of_driver'] = np.where(vehicle['journey_purpose_of_driver'].isin([6, 15]),
                                                        np.NaN,
                                                        vehicle['journey_purpose_of_driver'])

        # Convert `accident_reference` to object type, if not already, ensuring maintains leading zero if too few digits
        accident['accident_reference'] = accident_reference_fix(accident['accident_reference'])
        vehicle['accident_reference'] = accident_reference_fix(vehicle['accident_reference'])
        casualty['accident_reference'] = accident_reference_fix(casualty['accident_reference'])

        # Aggregate casualty data from person to vehicle
        casualty = vehicle[['accident_reference', 'vehicle_reference']].merge(casualty, how='left')  # include vehicles with no injuries
        casualty = aggregate_casualty_data(casualty)

        # Merge vehicle w/casualty info and accident info
        df_ = vehicle.merge(casualty, on=['accident_reference', 'vehicle_reference'], how='left')
        df_ = df_.merge(accident, on='accident_reference', how='inner')  # note - merge rate not quite 100%

        df = pd.concat([df, df_], axis=0)  # concat years together

    # Engineer datetime features
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.drop(columns=['date', 'time'], inplace=True)
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayw'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df['elapsed_time'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()  # total seconds since first timestamp

    # Sort final data
    df = df.sort_values(by=['accident_year', 'accident_reference', 'vehicle_reference']).reset_index(drop=True)

    # Convert all missings, -1, as np.NaN
    df = df.replace({-1: np.NaN})

    # Convert categorical features to categorical data type
    cats = categorical_features()
    numerics = numerical_features()
    df[cats] = df[cats].astype('category')
    df = df[cats + numerics + ['casualty_worst']]

    # Impute casualty modal type for vehicles with no casualties
    df = impute_casualty_modal_type(df)

    return df

def impute_casualty_modal_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes `casualty_modal_type` for vehicles missing any casualty information. Most cases are for vehicles where no
        casualty occurred. However, we still want to impute what object they might've struck.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    # Segment data into rows with missings versus without
    mi_cas_type = df[df['casualty_modal_type'].isnull()]
    df_ = df[~df.index.isin(mi_cas_type.index)]

    prediction_features = [i for i in df.columns if "casualty_modal_type" not in i]

    # Train val split
    X_train, X_val, y_train, y_val = train_test_split(df_[prediction_features],
                                                      df_['casualty_modal_type'],
                                                      test_size=0.2,
                                                      random_state=123,
                                                      stratify=df_['casualty_modal_type'],
                                                      shuffle=True)

    # Train classifier model with default params
    model = lgb.LGBMClassifier(objective="multiclass", random_state=123, n_estimators=100)
    model.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)], eval_metric='multi_error',
              callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

    # Generate predictions
    pred = model.predict(mi_cas_type[prediction_features])

    # Replace missing original values with predictions
    df.loc[df['casualty_modal_type'].isnull(), 'casualty_modal_type'] = pred

    return df

def accident_reference_fix(series: pd.Series) -> pd.Series:
    """
    Adds leading zeros to column `accident_reference`, which pd.read_csv might incorrectly read as int dtype column
    :param series: pd.Series
    :return: pd.Series
    """
    series = series.astype('O')  # if not already
    # Ensure all length 9
    return series.apply(lambda x: x.zfill(9))

def aggregate_casualty_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates casualty dataset from person-level up to vehicle-level. Creates new categories for vehicles with no
        casualties for fields ['casualty_class', 'casualty_severity']
    :param df: pd.DataFrame
    """
    # Insert new categories in ['casualty_class', 'casualty_severity'] for vehicles with no casualties
    df['casualty_class'] = np.where(df['casualty_class'].isna(), 0, df['casualty_class'])  # 0 = no casualty
    df['casualty_severity'] = np.where(df['casualty_severity'].isna(), 4, df['casualty_severity'])  # 4 = non-injury
    for col in ['casualty_class', 'casualty_severity']:
        df[col] = df[col].astype(int)

    # Casualty type, i.e. what did this vehicle hit (note - must explicitly exclude missings here)
    # Excludes vehicle passengers
    df['casualty_type'] = df['casualty_type'].replace(recode_vehicle_type())
    df['casualty_type'] = np.where(df['casualty_type'] == -1, np.NaN, df['casualty_type'])
    casualty_modal_type = df[(df['casualty_type'].notnull()) & (df['car_passenger'] == 0)]\
        .groupby(['accident_reference', 'vehicle_reference'])\
        ['casualty_type'].agg(lambda x: x.value_counts(dropna=True).index[0]).reset_index()\
        .rename(columns={'casualty_type': 'casualty_modal_type'})

    # Main variable to predict: worst casualty of vehicle, either in vehicle itself or pedestrian
    casualty_worst = df.groupby(['accident_reference', 'vehicle_reference'])['casualty_severity'].min().reset_index()\
        .rename(columns={'casualty_severity': 'casualty_worst'})

    df.drop(columns=['casualty_reference', 'casualty_class', 'sex_of_casualty', 'car_passenger',
                     'age_of_casualty', 'casualty_severity', 'casualty_type'], inplace=True)

    # Actual aggregation
    df = df.groupby(['accident_reference', 'vehicle_reference']).sum().reset_index()

    # Concat aggregated columns
    df = df.merge(casualty_worst, how='left')
    df = df.merge(casualty_modal_type, how='left')

    # Recode & consolidate casualty_worst down to 3 classes
    recode_casualty_worst = {
        4: 0,  # orig no injury
        3: 0,  # orig slight injury
        2: 1,  # orig severe injury
        1: 1  # orig fatality
    }
    df['casualty_worst'] = df['casualty_worst'].replace(recode_casualty_worst)

    # Fill missings, if any
    df = df.fillna(-1)

    return df

def cols_to_drop() -> Dict:
    """
    Dictionary of columns per dataset to drop.
    :return: dict
    """
    return {
        'accident':
            ['accident_index',
             'accident_year',
            'local_authority_ons_district',
            'location_northing_osgr',
            'location_easting_osgr',
            'police_force',  # focus on local_authority_district instead
            'accident_severity',  # Note - create own, don't want to use this as includes podestrians
            'local_authority_ons_district',  # seemingly redundant with local_authority_district
            'pedestrian_crossing_human_control',
            'pedestrian_crossing_physical_facilities',
            'did_police_officer_attend_scene_of_accident',  # endogenous to accident severity, so not relevant
            'lsoa_of_accident_location',  # surrogate for latitude/longitude
            'trunk_road_flag',
            'first_road_number',
            'second_road_number',
            'local_authority_highway',
            'day_of_week',
            'first_road_class',
            'second_road_class',
            'junction_detail',
            'number_of_casualties'],  # want to calculate own removing pedestrians

        'vehicle':
            ['accident_index',
            'age_band_of_driver',
            'vehicle_direction_from',
            'vehicle_direction_to',
            'generic_make_model',
            'lsoa_of_driver'],

        'casualty':
            ['accident_index',
             'accident_year',
            'pedestrian_location',
            'pedestrian_movement',
            'pedestrian_road_maintenance_worker',
            'casualty_imd_decile',
            'casualty_home_area_type',
            'lsoa_of_casualty',
            'bus_or_coach_passenger',
            'age_band_of_casualty']
    }

def drop_columns(df: pd.DataFrame, df_type: str) -> pd.DataFrame:
    """
    Drops select columns from input dataframe, `df`
    :param df: pd.DataFrame, input dataframe
    :param df_type: str, type of data in`df`
    :return: pd.DataFrame minus dropped columns
    """
    assert df_type in ['accident', 'vehicle', 'casualty']

    droppers = cols_to_drop()
    droppers = droppers[df_type]  # returns list

    return df[[i for i in df.columns if i not in droppers]]


def recode_vehicle_type() -> Dict:
    """
    Aggregates vehicle codes in column `vehicle_type` and `casualty_type` for lower dimensionality. Codes:
        -1: missing
        1: bicycle, e-scooters
        2: motorcycle, all types
        8: taxi/hire car
        9: car
        11: bus
        18: trams
        19: vans/goods vehicles
        90: other, including horse, agricultural
        np.NaN: (optional) applies to the case of `casualty_type` feature, since these are vehicles with no casualties
    :return: Dict
    """
    return {
        -1: -1,  # missing
        3: 2,  # all motorcycles are 2
        4: 2,
        5: 2,
        10: 11,  # all buses coded as 11
        16: 90,  # horses coded as other
        17: 90,  # agricultural vehicles to other
        20: 19,
        21: 19,
        22: 1,
        23: 2,
        97: 2,
        98: 19,
        99: -1
    }

def categorical_features() -> List:
    """
    Returns a list of categorical features in data
    :return: list
    """
    return [
                'vehicle_type',
                'casualty_modal_type',
                'towing_and_articulation',
                'vehicle_manoeuvre',
                'junction_location',
                'skidding_and_overturning',
                'hit_object_in_carriageway',
                'vehicle_leaving_carriageway',
                'vehicle_location_restricted_lane',
                'hit_object_off_carriageway',
                'first_point_of_impact',
                'vehicle_left_hand_drive',
                'journey_purpose_of_driver',
                'sex_of_driver',
                'propulsion_code',
                'driver_imd_decile',
                'driver_home_area_type',
                'local_authority_district',
                'road_type',
                'junction_control',
                'light_conditions',
                'weather_conditions',
                'road_surface_conditions',
                'special_conditions_at_site',
                'carriageway_hazards',
                'urban_or_rural_area',
                'dayw',
                'month',
                'hour',
                'accident_year'
            ]

def numerical_features() -> List:
    """
    Returns a list of numerical features in data
    :return: list
    """
    return [
                'day',
                'elapsed_time',
                'age_of_driver',
                'engine_capacity_cc',
                'age_of_vehicle',
                'longitude',
                'latitude',
                'number_of_vehicles',
                'speed_limit'
            ]