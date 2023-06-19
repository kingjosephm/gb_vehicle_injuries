import pandas as pd
from typing import Dict
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 150)


def read_data() -> pd.DataFrame:
    """

    :return:
    """
    for i in range(2019, 2022):

        # Read data & drop select columns
        vehicle = pd.read_csv(f"./data/dft-road-casualty-statistics-vehicle-{i}.csv", low_memory=False)
        vehicle = drop_columns(vehicle, 'vehicle')
        accident = pd.read_csv(f"./data/dft-road-casualty-statistics-accident-{i}.csv", low_memory=False)
        accident = drop_columns(accident, 'accident')
        casualty = pd.read_csv(f"./data/dft-road-casualty-statistics-casualty-{i}.csv", low_memory=False)
        casualty = drop_columns(casualty, 'casualty')

        # Verify missings all coded as '-1'
        assert vehicle.isnull().sum().sum() == 0
        assert casualty.isnull().sum().sum() == 0

        #######################
        ###### Data Mods  #####
        #######################

        # Recode missing latitude & longitude in accident dataset to -999
        for col in ['longitude', 'latitude']:
            accident[col] = np.where(accident[col].isnull(), -999, accident[col])

        # Recode vehicle type
        vehicle['vehicle_type'] = vehicle['vehicle_type'].replace(recode_vehicle_type())

        # Recode unknowns to missing
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

        # Aggregate casualty data from person to vehicle
        casualty = aggregate_casualty_data(casualty)

        # Merge vehicle w/casualty info
        df = vehicle.merge(casualty, on=['accident_reference', 'vehicle_reference'], how='left')

        df = df.iloc[:, :].merge(accident, on='accident_reference', how='left')


    '''
    # Verify:
        number_of_vehicles in accident == unique vehicles per accident_reference
        we have casualty_severity for each vehicle
        numbers align between datasets
        casualty_class ~= casualty_type
    '''

    return df

def aggregate_casualty_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates casualty dataset from person-level up to vehicle-level
    :param df:
    """
    # Share of males among casualties, non-missing only
    df['sex_of_casualty'] = np.where(df['sex_of_casualty'] == 9, -1, df['sex_of_casualty'])
    df['sex_of_casualty'].replace({2: 0}, inplace=True)
    casualty_share_male = df[df['sex_of_casualty'] != -1].groupby(['accident_reference', 'vehicle_reference'])\
        ['sex_of_casualty'].mean().reset_index().rename(columns={'sex_of_casualty': 'casualty_share_male'})

    # Mean age across casualties, non-missing only
    casualty_mean_age = df[df['age_of_casualty'] != -1].groupby(['accident_reference', 'vehicle_reference'])\
        ['age_of_casualty'].mean().reset_index().rename(columns={'age_of_casualty': 'casualty_mean_age'})

    # Main variable to predict: worst casualty of vehicle, either in vehicle itself or pedestrian
    casualty_worst = df.groupby(['accident_reference', 'vehicle_reference'])['casualty_severity'].min().reset_index()\
        .rename(columns={'casualty_severity': 'casualty_worst'})

    # One-hot encode categorical variables to aggregate, yields total number of each categorical variable per vehicle
    dummies = pd.DataFrame()
    for col in ['casualty_class', 'casualty_severity']:
        dum = pd.get_dummies(df[col],
                             prefix=col)  # note - verified above no missings, so no need to create separate dummies for them
        dummies = pd.concat([dummies, dum], axis=1)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=['casualty_reference', 'casualty_class', 'sex_of_casualty', 'age_of_casualty', 'casualty_severity'], inplace=True)

    # Actual aggregation
    df = df.groupby(['accident_reference', 'vehicle_reference']).sum().reset_index()

    # Concat aggregated columns
    df = df.merge(casualty_worst, how='left')
    df = df.merge(casualty_share_male, how='left')
    df = df.merge(casualty_mean_age, how='left')

    # Fill missings, if any
    df = df.fillna(-1)

    # Total casualties
    # Total casualties
    df['casualty_total'] = df[[i for i in df.columns if 'casualty_severity' in i]].sum(axis=1)

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
            'second_road_number'
            'number_of_casualties'],  # want to calculate own removing pedestrians

        'vehicle':
            ['accident_index',
            'age_band_of_driver',
            'vehicle_direction_from',
            'vehicle_direction_to',
            'generic_make_model'],

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
            'casualty_type',
            'car_passenger',
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
    Aggregates some vehicle codes in column `vehicle_type` for lower dimensionality. Codes:
        -1: missing
        1: bicycle, e-scooters
        2: motorcycle, all types
        8: taxi/hire car
        9: car
        11: bus
        18: trams
        19: vans/goods vehicles
        90: other, including horse, agricultural
    :return: Dict
    """
    return {
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

if __name__ == '__main__':
    read_data()