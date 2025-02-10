import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import numpy as np


def scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to return Z scored
    data for PCA

    Parameters
    ----------
    df: np.array
        Matrix of values

    Returns
    -------
    pd.DataFrame:
        DataFrame of scaled values
    """

    removed_data = []
    for col in df.columns:
        if df[col].dtype != "float":
            removed_data.append(col)
            scaled_df = df.drop(col, axis=1)

    scaled_data = StandardScaler().fit_transform(scaled_df)
    scaled_data = pd.DataFrame(
        scaled_data,
        columns=[col for col in df.columns.to_list() if col not in removed_data],
    )
    scaled_data["sex"] = df["sex"].reset_index(drop=True)
    return scaled_data


def PCA_analysis(data: np.array) -> object:
    """
    Function to do PCA

    Parameters
    ----------
    data: np.array
        array of data to do PCA on

    Returns
    -------
    decomp: object
       PCA model

    """
    return PCA().fit(data)


def permutation_null_distro(data: pd.DataFrame, n_perms: int = 5000) -> np.array:
    """
    Function to permute the null distribution

    Parameters
    ----------
    data: pd.DataFrame
        data to permuate

    n_perms: int=5000
        number of permuations

    Returns
    -------
    explained_variance_perm: np.array
        array of null distribution for each
        component
    """
    explained_variance_perm = np.zeros((n_perms, data.shape[1]))
    for perm in range(n_perms):
        perm_data = data.copy()
        for col in range(data.shape[1]):
            perm_data.iloc[:, col] = np.random.permutation(perm_data.iloc[:, col])
        perm_data = scaling(perm_data)
        pca_perm = PCA_analysis(perm_data)
        explained_variance_perm[perm] = pca_perm.explained_variance_ratio_
    return explained_variance_perm


def get_crit_val(number_of_components: int, null_distro: np.array) -> dict:
    """
    Function to determine crit val

    Parameters
    ----------
    number_of_components: int
        number of components to check
    null_distro: np.array
        array of the null distibution

    Returns
    -------
    crti_val: dict
        dictionary of criticial values
    """
    crit_val = {}
    for comp in range(number_of_components):
        null_distribution = null_distro[:, comp]
        if max(null_distribution) > 0 and min(null_distribution) < 0:
            crit_val[comp] = np.abs(np.quantile(null_distribution, 0.975))
        if min(null_distribution) > 0:
            crit_val[comp] = np.quantile(null_distribution, 0.95)
        if max(null_distribution) <= 0:
            crit_val[comp] = np.quantile(null_distribution, 0.05)
    return crit_val


def get_explained_ratio(alt_pca: object, n_comp: int):
    """
    Function to organise the explained ratio for
    comparison

    Parameters
    ----------
    alt_pca: object
        fitted sklearn.decomposition.PCA object
    n_comp: int
        number of components
    """
    ratio = {}
    for comp in range(n_comp):
        ratio[comp] = alt_pca.explained_variance_ratio_[comp]
    return ratio


def get_significant_components(crti_val: dict, alt_val: dict) -> list:
    """
    Funciton to get significant components

    Parameters
    ----------
    crti_val: dict
        dictionary of criticial values
    alt_val: dict

    Returns
    -------

    """
    components = []
    for comp in crti_val.keys():
        if alt_val[comp] > crti_val[comp]:
            components.append(comp)
    return components


def create_dataframe(data: pd.DataFrame, key: pd.DataFrame, *args: str):
    """
    Function to create dataframe of HCP data given
    data and key

    Parameters
    ----------
    data: pd.DataFrame
        dataframe with 'src_subject_id', 'interview_date',
        'interview_age',
    key: pd.DataFrame
         Dataframe with 'src_subject_id',
         'interview_age', 'sex', 'phenotype'
    args: str
        str of column(s) to include in the final
        dataframe
    """
    merged = (
        pd.merge(
            data[["src_subject_id", "interview_date", "interview_age"] + list(args)],
            key,
            on="src_subject_id",
            how="right",
        )
        .sort_values(by="src_subject_id")
        .reset_index(drop=True)
    )
    null_values = (
        merged[merged["interview_date"].isna()]
        .drop("interview_age_x", axis=1)
        .rename(columns={"interview_age_y": "interview_age"})
    )
    beh_data = merged[
        merged["interview_age_x"].astype("float")
        == merged["interview_age_y"].astype("float")
    ].reset_index(drop=True)
    duplicated = beh_data[beh_data["src_subject_id"].duplicated()]
    beh_data = (
        beh_data.drop(duplicated.index)
        .drop("interview_age_x", axis=1)
        .rename(columns={"interview_age_y": "interview_age"})
    )
    return (
        pd.concat([beh_data, null_values], axis=0)
        .sort_values(by="phenotype")
        .drop("interview_date", axis=1)
        .reset_index(drop=True)
    )


def check_data(data: pd.Series, key: pd.Series) -> list:
    """
    Function to see if any elements
    are missing between data and key

    Parameters
    ----------
    data: pd.Series
        A Series
    key: pd.Series

    Returns
    -------
    list: list object
        list of missing subjects
        between key and data

    """
    return [
        el
        for el in key["src_subject_id"].to_list()
        if el not in data["src_subject_id"].to_list()
    ]


def get_missing_data_from_key(data: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: make more dynamic so its not only the first one
    """
    missing = check_data(data, key)
    if not missing:
        raise ValueError("No missing values")
    return (
        pd.concat([data, key[key["src_subject_id"] == missing[0]]], axis=0)
        .sort_values(by="phenotype")
        .reset_index(drop=True)
    )


def imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to impute dataframe by
    KMNIImputer

    Parameters
    ---------
    df: pd.DataFrame
        DataFrame

    Returns
    -------
    pd.DataFrame: imputed DataFrame
    """
    imputed_data = {}
    groups = df.groupby("phenotype")
    for group in groups.all().index:
        group_df = groups.get_group(group)
        imputed = KNNImputer().fit_transform(group_df[group_df.columns[4:]].values)
        data = pd.concat(
            [
                group_df[
                    ["src_subject_id", "phenotype", "interview_age", "sex"]
                ].reset_index(drop=True),
                pd.DataFrame(imputed).reset_index(drop=True),
            ],
            axis=1,
        )
        data.columns = group_df.columns
        imputed_data[group] = data
    return pd.concat(imputed_data.values())
