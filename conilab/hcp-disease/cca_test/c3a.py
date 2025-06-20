import numpy as np
from scipy.optimize import minimize
from itertools import combinations


class C3A:
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

    def fit(self, *data_sets: tuple) -> None:
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
        self._calculate_covariance_matricies(*data_sets)
        self._get_dimensions(*data_sets)
        self._weight_intialization()
        self._optimise()

    def transform(self, *data_sets: tuple) -> list[np.ndarray]:
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
        assert (
            self.weights_ is not None
        ), "Model must be fitted before transform can be called."
        assert len(data_sets) == len(
            self.dims_
        ), "Model fitted with different number of datasets."

        self.projections_ = [
            np.stack(
                [
                    self._normalise(self._normalise(X_data) @ wx)
                    if self.normalise_weights
                    else X_data @ wx,
                    self._normalise(self._normalise(Y_data) @ wb)
                    if self.normalise_weights
                    else Y_data @ wb,
                ],
                axis=0,
            )
            for (X_data, Y_data), (wx, wb) in zip(data_sets, self.weights_)
        ]

        self.canonical_correlations_ = [
            np.corrcoef(data_sets[0], data_sets[1])[0, 1]
            for data_sets in self.projections_
        ]
        return self.projections_

    def fit_transform(self, *data_sets) -> list[np.ndarray]:
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
        self.fit(*data_sets)
        return self.transform(*data_sets)

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
        assert (
            self.canonical_correlations_ is not None
        ), "Model must be fitted and transfomed before correlations can be returned"
        return self.canonical_correlations_

    def compute_loadings(
        self, *data_sets: tuple
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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
        assert (
            self.projections_ is not None
        ), "Model must be fitted and transfomed before computing loadings."
        return [
            (
                np.corrcoef(self._normalise(X_data).T, x_proj, rowvar=True)[:-1, -1],
                np.corrcoef(self._normalise(Y_data).T, y_proj, rowvar=True)[:-1, -1],
            )
            for (X_data, Y_data), (x_proj, y_proj) in zip(data_sets, self.projections_)
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
        init_weights = []

        for idx, _ in enumerate(self.dims_):
            s_xb = self.covariances_[f"s_X{idx+1}_Y{idx+1}"]
            # Perform SVD on the cross-covariance matrix
            try:
                U, _, Vt = np.linalg.svd(s_xb, full_matrices=False)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"SVD failed for dataset {idx+1}: {e}")

            wx = U[:, 0]
            wb = Vt.T[:, 0]
            s_xx = self.covariances_[f"s_X{idx+1}_X{idx+1}"]
            s_bb = self.covariances_[f"s_Y{idx+1}_Y{idx+1}"]

            wx = wx / np.sqrt(wx.T @ s_xx @ wx)
            wb = wb / np.sqrt(wb.T @ s_bb @ wb)

            init_weights.extend(wx)
            init_weights.extend(wb)

        self.intial_weights_ = np.array(init_weights)

    def _calculate_covariance_matricies(self, *data_sets) -> dict:
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
        for idx, study_pair in enumerate(data_sets):
            X_data, Y_data = study_pair
            self._data_able_to_process(study_pair)
            X_data = self._normalise(X_data)
            Y_data = self._normalise(Y_data)
            study_num = idx + 1
            try:
                self.covariances_[f"s_Y{study_num}_Y{study_num}"] = (
                    self._create_covariance_matrix(Y_data, Y_data)
                )
                self.covariances_[f"s_X{study_num}_X{study_num}"] = (
                    self._create_covariance_matrix(X_data, X_data)
                )
                self.covariances_[f"s_X{study_num}_Y{study_num}"] = (
                    self._create_covariance_matrix(X_data, Y_data)
                )

            except Exception as e:
                print(f"Error calculating covariances for Study {study_num}: {e}")

    def _data_able_to_process(self, study_pair: tuple) -> bool:
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
        assert (
            isinstance(study_pair, (tuple, list)) and len(study_pair) == 2
        ), "Given argument isn't a pair of datasets"
        assert isinstance(study_pair[0], np.ndarray) or not isinstance(
            study_pair[1], np.ndarray
        ), "Data provided ins't numpy array"
        assert (study_pair[0].shape[0] != 0) and (
            study_pair[1].shape[0] != 0
        ), "Study pairs contains not data"
        assert (
            study_pair[0].shape[0] == study_pair[1].shape[0]
        ), f"Mismatch between ({study_pair[0].shape[0]} and {study_pair[1].shape[0]})"

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

    def _get_dimensions(self, *data_sets) -> None:
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
        self.dims_ = [(X.shape[1], Y.shape[1]) for X, Y in data_sets]

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
        offset = 0
        split_weights = []
        for X_dim, Y_dim in self.dims_:
            wx = weights[offset : offset + X_dim]
            offset += X_dim
            wb = weights[offset : offset + Y_dim]
            offset += Y_dim
            split_weights.append((wx, wb))
        return split_weights

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
        weights_ = self._split_weights(weights)
        for idx, (wx, wb) in enumerate(weights_):
            s_xb = covariances[f"s_X{idx+1}_Y{idx+1}"]
            s_xx = covariances[f"s_X{idx+1}_X{idx+1}"]
            s_bb = covariances[f"s_Y{idx+1}_Y{idx+1}"]
            total_loss += self._cross_cov_term(wb, s_xb, wx)
            total_loss += self._regularization_term(wx, s_xx, l2)
            total_loss += self._regularization_term(wb, s_bb, l2)

        # Similarity penalty across imaging weights
        if theta > 0 and len(weights_) > 1:
            total_loss += sum(
                self._dissimilarity_penality(theta, w1[0], w2[0])
                for w1, w2 in combinations(weights_, 2)
            )

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

    def _score(self, *data_sets: tuple) -> float:
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

        self.transform(*data_sets)
        correlations = self.calculate_canonical_correlations()
        return np.mean(correlations)


class GridSearchCA3:
    def __init__(self, l2_values, theta=0, tol=1e-6, maxiter=500, verbose=False):
        """
        Custom grid search to find the best l2 value for the CA3 model.

        Parameters
        ----------
        l2_values : list of float
            The l2 regularization parameters to search over.
        theta : float
            The dissimilarity regularization parameter (shared across all models).
        tol : float
            Tolerance for optimization.
        maxiter : int
            Maximum number of optimization iterations.
        verbose : bool
            If True, print progress during search.
        """
        self.l2_values = l2_values
        self.theta = theta
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.best_model_ = None
        self.best_score_ = -np.inf
        self.best_l2_ = None
        self.all_results_ = []

    def fit(self, *data_sets):
        """
        Fit CA3 models with each l2 value and track the one with best score.
        """
        for l2 in self.l2_values:
            model = CA3(l2=l2, theta=self.theta, tol=self.tol, maxiter=self.maxiter)
            model.fit(*data_sets)
            score = model._score(*data_sets)

            self.all_results_.append((l2, score))

            if self.verbose:
                print(f"l2: {l2:.4f}, score: {score:.4f}")

            if score > self.best_score_:
                self.best_score_ = score
                self.best_model_ = model
                self.best_l2_ = l2

        if self.verbose:
            print(f"Best l2: {self.best_l2_:.4f}, Best score: {self.best_score_:.4f}")

    def get_best_model(self):
        return self.best_model_

    def get_best_l2(self):
        return self.best_l2_

    def get_all_results(self):
        return self.all_results_
