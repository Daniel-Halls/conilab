import pandas as pd


def reorganise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to re-organise a dataframe by max row
    and max column

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of values

    Returns
    -------
    df: pd.Dataframe
        re-organised dataframe
    """
    df["max"] = df.max(axis=1)
    df = df.sort_values(by="max", ascending=False).drop(columns=["max"])
    max_col = df.max()
    sorted_columns = max_col.sort_values(ascending=False).index
    return df[sorted_columns]


def create_correlation_table(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Function to get correlation table

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of correlation values
    col1: str
        name of column of tracts
    col2: str
        name of columns of r2 values

    """
    max_indices = df.idxmax()
    max_values = {
        col: [max_indices[col], df[col].loc[max_indices[col]]] for col in df.columns
    }
    return pd.DataFrame(max_values).T.rename(columns={0: col1, 1: col2})


def organise_keys(dictionary: dict) -> dict:
    """
    Function to organise keys of
    a dictionary

    Parameters
    ----------
    dictionary: dict
       dictionary

    Returns
    -------
    dict: dictionary
        orgnaised dict
    """
    return {key: dictionary[key] for key in sorted(dictionary.keys())}
