import pandas as pd


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
