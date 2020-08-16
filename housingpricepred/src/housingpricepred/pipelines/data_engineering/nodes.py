import pandas as pd


def prepocess_trainingdata(housingdata_train: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for companies.

        Args:
            housingdata_train: Source data.
        Returns:
            Preprocessed data.

    """

    featureList = [
            "LotFrontage",
            "LotArea",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal"
        ]

    for elem in featureList:
        housingdata_train[elem] = housingdata_train[elem].fillna(0)

    target = ["SalePrice"]

    return housingdata_train[featureList+target]