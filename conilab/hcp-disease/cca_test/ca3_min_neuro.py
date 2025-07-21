import numpy as np
from scipy.optimize import minimize


class C3A_minimize_neuroimaging:
    """
    C3A class.
    A class to do C3A

    Usage
    -----
    c3a = C3A(l2=0.5, theta=1)
    c3a.fit(study1, study2)
    transformed = c3a.transform(study1, study2)
    """

    def __init__(
        self,
        l2: float = 1,
        theta: float = 0,
        tol=1e-6,
        maxiter=500,
        normalise_weights=True,
    ):
        self.l2_ = l2
        self.theta_ = theta
        self.intial_weights_ = None
        self.dims_ = []
        self.best_loss = float("inf")
        self.weights_ = None
        self.covariances_ = {}
        self.tol_ = tol
        self.maxiter_ = maxiter
        self.normalise_weights = normalise_weights
        self.canonical_correlations_ = None
        self.projections_ = None

    def fit(self, X, Y) -> None:
        """
        Method to fit the CA3 model to a given
        set of datasets

        Parameters
        ----------
        data_sets: tuple
            a tuple of X, Y data
            from an arbituray number of
            datasets

        Returns
        -------
        None
        """
        self._calculate_covariance_matricies(X, Y)
        self._get_dimensions(X, Y)
        self._weight_intialization()
        self._optimise()

    def transform(self, X, Y: tuple) -> list[np.ndarray]:
        """
        Methods to transform data sets into canonical
        projects.

        Parameters
        ----------
        data_sets: tuple
            a tuple of X, Y data
            from an arbituray number of
            datasets

        Returns
        --------
        projects: list[np.ndarray]
            conatins a list of the
            projections of each dataset in
            ndarry of n_components by n_samples
        """
        assert self.weights_ is not None, (
            "Model must be fitted before transform can be called."
        )

        x_projections = self._normalise(self._normalise(X) @ self.weights_[0])
        y_projections = self._normalise(self._normalise(Y) @ self.weights_[1])
        self.projections_ = np.stack([x_projections, y_projections])
        self.canonical_correlations_ = np.corrcoef(x_projections, y_projections)[0, 1]
        return self.projections_

    def fit_transform(self, X, Y) -> list[np.ndarray]:
        """
        Methods to fit a CA3 model and then transform
        the data.

        Parameters
        ----------
        data_sets: tuple
            a tuple of X, Y data
            from an arbituray number of
            datasets

        Returns
        --------
        projects: list[np.ndarray]
            conatins a list of the
            projections of each dataset in
            ndarry of n_components by n_samples.
        """
        self.fit(X, Y)
        return self.transform(X, Y)

    def calculate_canonical_correlations(self) -> list[float]:
        """
        Method to obtain the canonical correlations.
        Model must have been fitted and transfomed
        before.

        Parameters
        ----------
        None

        Returns
        -------
        canonical_correlations: list[float]
            list of canonical correlations
        """
        assert self.canonical_correlations_ is not None, (
            "Model must be fitted and transfomed before correlations can be returned"
        )
        return self.canonical_correlations_

    def compute_loadings(self, X, Y) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Computes canonical loadings for each study.

        Parameters
        ----------
        data_sets: tuple
            List of (img_data, beh_data) pairs.

        Returns
        -------
        loadings: list of tuples
            Each tuple contains (img_loadings, beh_loadings), i.e., correlations between
            original features and their respective canonical variates.
        """
        assert self.projections_ is not None, (
            "Model must be fitted and transfomed before computing loadings."
        )
        return [
            np.corrcoef(self._normalise(X).T, self.projections_[0], rowvar=True)[
                :-1, -1
            ],
            np.corrcoef(self._normalise(Y).T, self.projections_[1], rowvar=True)[
                :-1, -1
            ],
        ]

    def _weight_intialization(self) -> np.ndarray:
        """
        Method to define a set of random starting
        weights

        Parameters
        ----------
        weights: tuple(int)
            tuple of set amount
            of int values

        Returns
        -------
        np.ndarrray
            array of numpy values
        """

        s_xb = self.covariances_["s_X_Y"]
        try:
            U, _, Vt = np.linalg.svd(s_xb, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed due to: {e}")

        wx = U[:, 0]
        wb = Vt.T[:, 0]
        s_xx = self.covariances_["s_X_X"]
        s_bb = self.covariances_["s_Y_Y"]

        wx = wx / np.sqrt(wx.T @ s_xx @ wx)
        wb = wb / np.sqrt(wb.T @ s_bb @ wb)
        self.intial_weights_ = np.concat([wx, wb])

    def _calculate_covariance_matricies(self, X_data, Y_data) -> dict:
        """
        Calculates covariance and auto covariance
        matricies

        Parameters
        ----------
        study_pairs: tuple
            a tuple or list containing two numpy arrays:
            (behavioural_data, imaging_data).
            Assumes data is (subjects x features).

        Returns
        -------
        covariance_results: dict
            dictionary of covariance and auto-covariance matrices

        """
        self._data_able_to_process(X_data, Y_data)
        X_data = self._normalise(X_data)
        Y_data = self._normalise(Y_data)

        try:
            self.covariances_["s_Y_Y"] = self._create_covariance_matrix(Y_data, Y_data)
            self.covariances_["s_X_X"] = self._create_covariance_matrix(X_data, X_data)
            self.covariances_["s_X_Y"] = self._create_covariance_matrix(X_data, Y_data)
        except Exception as e:
            print(f"Error calculating covariances due to: {e}")

    def _data_able_to_process(self, X_data, Y_data) -> bool:
        """
        Method to check that data
        is in correct format to be processed

        Parameters
        ----------
         study_pair: tuple,
             tuple of behavioural data
             and imging data


        Returns
        -------
        bool: boolean
            bool of if failed or not
        """
        assert isinstance(Y_data, np.ndarray) or not isinstance(X_data, np.ndarray), (
            "Data provided ins't numpy array"
        )
        assert (X_data.shape[0] != 0) and (Y_data.shape[0] != 0), (
            "Study pairs contains not data"
        )
        assert X_data.shape[0] == Y_data.shape[0], (
            f"Mismatch between ({X_data.shape[0]} and {Y_data.shape[0]})"
        )

    def _optimise(self) -> None:
        """
        Method to minimise the
        objective function

        Parameters
        ----------
        None

        Returns
        --------
        None
        """
        model = minimize(
            self._objective_function,
            self.intial_weights_,
            options={"gtol": self.tol_, "maxiter": self.maxiter_},
            args=(self.covariances_, self.theta_, self.l2_),
        )
        self.best_loss = model.fun
        self.weights_ = self._split_weights(model.x)

    def _get_dimensions(self, X, Y) -> None:
        """
        Method to get the dimensions
        of the data

        Parameters
        ----------
        *data_sets: tuple
            tuple of datasets

        Returns
        -------
        None
        """
        self.dims_ = [X.shape[1], Y.shape[1]]

    def _split_weights(self, weights: np.ndarray) -> list[np.ndarray]:
        """
        Splits the flat weight vector weights into individual vectors
        for each x and b dataset.

        Parameters
        ----------
        weights: np.ndarray
            flatten numpy array

        Returns
        -------
        split_weights: list[np.ndarray]
            list of weights split
            wx and wb

        """
        wx = weights[0 : self.dims_[0]]
        wb = weights[self.dims_[0] : self.dims_[1] + 1]
        return [wx, wb]

    def _objective_function(
        self, weights: np.ndarray, covariances: dict, theta: float, l2: float
    ) -> float:
        """
        Objective function of the CA3 class

        Parameters
        ----------
        weights: np.ndarray
            weights
        covariances: dict
            dict of cross/auto covariance
            matricies
        theta: float
            theta penality
        l2: float
            regularization penailty

        Returns
        -------
        total_loss: float
           total loss of the objective function
        """
        total_loss = 0
        wx, wb = self._split_weights(weights)
        s_xb = covariances["s_X_Y"]
        s_xx = covariances["s_X_X"]
        s_bb = covariances["s_Y_Y"]
        total_loss += self._cross_cov_term(wb, s_xb, wx)
        total_loss += self._regularization_term(wx, s_xx, l2)
        total_loss += self._regularization_term(wb, s_bb, l2)

        ## Similarity penalty across imaging weights, this needs changing
        # if theta > 0 and len(weights_) > 1:
        #    total_loss += sum(
        #        self._dissimilarity_penality(theta, w1[0], w2[0])
        #        for w1, w2 in combinations(weights_, 2)
        #    )

        return total_loss

    def _create_covariance_matrix(
        self, matrix_1: np.ndarray, matrix_2: np.ndarray
    ) -> np.ndarray:
        """
        Function to calculate cross-auto
        covariance matrix

        Parameters
        ----------
        matrix_1: np.ndarray
            A matrix tht should
            correspond to subject by
            features
        matrix_2: np.ndarray
            A matrix that should
            correspond to features by
            feautres

        Returns
        -------
        np.ndarray: array
            array of covariance matrix
        """
        return (matrix_1.T @ matrix_2) / matrix_1.shape[0]

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        """
        Function to normalise data.

        Parmeteres
        ----------
        data: np.ndarray
            data to demean

        Returns
        -------
        np.ndarray: array
            demeaned data
        """
        dmean = data - data.mean(axis=0)
        std = data.std(axis=0, ddof=1)
        std = np.where(std == 0.0, 1.0, std)
        return dmean / std

    def _cross_cov_term(
        self, weight_Y: np.ndarray, cov_mat: np.ndarray, weight_X: np.ndarray
    ) -> np.ndarray:
        """
        Method to calculate the cross covarance term
        in the objective function

        Parameters
        ----------
        weight_Y: np.ndarray
            set of weights for wb
        cov_mat: np.ndarray
             covariance matrix for
             wx wb
        weight_X: np.ndarray
            set of weights for wx

        Returns
        -------
        np.ndarray: np.array
            cross covariance term
        """
        return -weight_X.T @ (cov_mat @ weight_Y)

    def _regularization_term(
        self, weight: np.ndarray, cov_mat: np.ndarray, lambda_i: float
    ) -> float:
        """
        Method to calculate the regularization term
        in the objective function

        Parameters
        ----------
        weight: np.ndarray
            set of weights
        cov_mat: np.ndarray
            auto covariance matrix
        lambda_i: float
            regularization parameter

        Returns
        -------
        float: float
            regularization term of the objective function
        """
        return 0.5 * lambda_i * (weight.T @ (cov_mat @ weight) - 1)

    def _dissimilarity_penality(
        self, theta_r: float, X_weight1: np.ndarray, X_weight2: np.ndarray
    ) -> float:
        """
        Method to return dissimilarity penality

        Parameters
        -----------
        theta_r: float
           theta penality.
        img_weight1: np.ndarray
            weights of imaging data
        img_weight2: np.ndarray
            weights of second imaging
            data

        Returns
        -------
        float: float
            dissimilarity penality
        """
        return theta_r * 0.5 * np.sum((X_weight1 - X_weight2) ** 2)

    def _score(self, X, Y) -> float:
        """
        Method used to evaluate model performance.

        Parameters
        -----------
        data_sets: tuple
            a tuple of X, Y data
            from an arbituray number of
            datasets

        Returns
        -------
        float: float
            mean of correlation
            values across datasets

        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before scoring.")

        self.transform(X, Y)
        correlations = self.calculate_canonical_correlations()
        return np.mean(correlations)
