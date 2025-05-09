class CA3:
    def __init__(
        self,
        l2: float = None,
        theta: float = None,
        random_seed: int = 42,
        tol=1e-6,
        maxiter=500,
    ):
        self.l2_ = np.logspace(-6, -2, num=5) if l2 is None else l2
        self.theta_ = np.arange(0.0, 1, 0.1) if theta is None else theta
        self.intial_weights_ = None
        self.dims_ = []
        self.best_loss = float("inf")
        self.optimal_theta_r = None
        self.optimal_l2 = None
        self.weights_ = None
        self.covariances_ = {}
        self.tol_ = tol
        self.maxiter_ = maxiter
        self.rng = np.random.RandomState(random_seed)

    def fit(self, *data_sets):
        self._calculate_covariance_matricies(*data_sets)
        self._get_dimensions(*data_sets)
        self._weight_intialization(*list(itertools.chain(*self.dims_)))
        self._optimise()

    def transform(self, *data_sets):
        assert (
            self.weights_ is not None
        ), "Model must be fitted before transfomed can be called."
        assert len(data_sets) == len(
            self.dims_
        ), "Model fitted with different number of datasets."

        scores = {}
        correlations = {}
        count = 0
        for (img_data, beh_data), (wx, wb) in zip(data_sets, self.weights_):
            imging_projections = img_data @ wx
            beh_projections = beh_data @ wb
            scores[f"study{count}"] = [imging_projections, beh_projections]
            corr = np.array([np.corrcoef(imging_projections, beh_projections)[0, 1]])
            correlations[f"study{count}"] = corr
            count += 1

        return {"correlations": correlations, "projections": scores}

    def fit_transform(self, *data_sets):
        self.fit(*data_sets)
        return self.transform(*data_sets)

    def _weight_intialization(self, *dims) -> np.ndarray:
        """
        Define a set of random starting
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

        for idx, (behav_dim, img_dim) in enumerate(self.dims_):
            s_xb = self.covariances_[f"s_img{idx+1}_behav{idx+1}"]

            # Perform SVD on the cross-covariance matrix
            try:
                U, S, Vt = np.linalg.svd(s_xb, full_matrices=False)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"SVD failed for dataset {idx+1}: {e}")

            # First left/right singular vectors as initial directions
            wx = U[:, 0]
            wb = Vt.T[:, 0]

            # Normalize to unit norm under their autocovariance
            s_xx = self.covariances_[f"s_img{idx+1}_img{idx+1}"]
            s_bb = self.covariances_[f"s_behav{idx+1}_behav{idx+1}"]

            wx = wx / np.sqrt(wx.T @ s_xx @ wx + 1e-8)
            wb = wb / np.sqrt(wb.T @ s_bb @ wb + 1e-8)

            init_weights.extend(wx)
            init_weights.extend(wb)

        self.intial_weights_ = np.array(init_weights)

    def _calculate_covariance_matricies(self, *data_sets) -> dict:
        """
        Calculates covariance matrices and auto covariance
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
            img_data, behav_data = study_pair
            if not self._data_able_to_process(study_pair, behav_data, img_data):
                continue
            behav_data = self._mean_center(behav_data)
            img_data = self._mean_center(img_data)
            study_num = idx + 1
            try:
                self.covariances_[f"s_behav{study_num}_behav{study_num}"] = (
                    self._create_covariance_matrix(behav_data, behav_data)
                )
                self.covariances_[f"s_img{study_num}_img{study_num}"] = (
                    self._create_covariance_matrix(img_data, img_data)
                )
                self.covariances_[f"s_img{study_num}_behav{study_num}"] = (
                    self._create_covariance_matrix(img_data, behav_data)
                )

            except Exception as e:
                print(f"Error calculating covariances for Study {study_num}: {e}")

    def _data_able_to_process(
        self, study_pair: tuple, behav_data: np.ndarray, img_data: np.ndarray
    ) -> bool:
        """
        Function to check that data
        is in correct format to be processed

        Parameters
        ----------
         study_pair: tuple,
             tuple of behavioural data
             and imging data
         behav_data: np.ndarray
             array of behav_data
         img_data: np.ndarray
             array of img_data

        Returns
        -------
        bool: boolean
            bool of if failed or not
        """
        if not isinstance(study_pair, (tuple, list)) or len(study_pair) != 2:
            print("Given argument isn't a pair of datasets")
            return False
        if not isinstance(behav_data, np.ndarray) or not isinstance(
            img_data, np.ndarray
        ):
            print("Data provided isn't a numpy array")
            return False
        if (
            behav_data.shape[0] == 0
            or img_data.shape[0] == 0
            or behav_data.shape[0] != img_data.shape[0]
        ):
            print(f"Mismatch between ({behav_data.shape[0]} and {img_data.shape[0]})")

        return True

    def _optimise(self):
        if isinstance(self.theta_, (list, np.ndarray)):
            for theta in self.theta_:
                model = self._optimising_model(theta)
                if model.status != 0:
                    continue
                if model.fun < self.best_loss:
                    self.best_loss = model.fun
                    self.optimal_theta_r = theta
                    self.weights_ = self._split_weights(model.x)
        else:
            model = self._optimising_model(self.theta_)
            self.best_loss = model.fun
            self.optimal_theta_r = self.theta_
            self.weights_ = self._split_weights(model.x)

        if isinstance(self.l2_, (list, np.ndarray)):
            for l2 in self.l2_:
                model = self._optimising_model(self.optimal_theta_r, l2)
                if model.status != 0:
                    print("model failed")
                    continue
                if model.fun < self.best_loss:
                    print("l2 succeeded")
                    self.best_loss = model.fun
                    self.optimal_l2 = l2
                    self.weights_ = self._split_weights(model.x)
        else:
            model = self._optimising_model(self.optimal_theta_r, self.l2_)
            self.best_loss = model.fun
            self.optimal_l2 = self.l2_
            self.weights_ = self._split_weights(model.x)

    def _optimising_model(self, theta, l2=1):
        return minimize(
            self._objective_function,
            self.intial_weights_,
            options={"gtol": self.tol_, "maxiter": self.maxiter_},
            args=(self.covariances_, theta, l2),
            method="L-BFGS-B",
        )

    def _get_dimensions(self, *data_sets):
        self.dims_ = [(behav.shape[1], img.shape[1]) for behav, img in data_sets]

    def _split_weights(self, w):
        """
        Splits the flat weight vector w into individual vectors
        for each behavioural and imaging dataset.
        """
        offset = 0
        weights = []
        for img_dim, behav_dim in self.dims_:
            wx = w[offset : offset + img_dim]
            offset += img_dim
            wb = w[offset : offset + behav_dim]
            offset += behav_dim
            weights.append((wx, wb))
        return weights

    def _objective_function(self, weights, covariances, theta, l2):
        total_loss = 0
        weights_ = self._split_weights(weights)
        for idx, (wx, wb) in enumerate(weights_):
            s_xb = covariances[f"s_img{idx+1}_behav{idx+1}"]
            s_xx = covariances[f"s_img{idx+1}_img{idx+1}"]
            s_bb = covariances[f"s_behav{idx+1}_behav{idx+1}"]
            total_loss += self._cross_cov_term(wb, s_xb, wx)
            total_loss += self._regularization_term(wx, s_xx, l2)
            total_loss += self._regularization_term(wb, s_bb, l2)

        # Similarity penalty across imaging weights
        if theta > 0 and len(weights_) > 1:
            for img_data in range(len(weights_)):
                for next_img_data in range(img_data + 1, len(weights_)):
                    total_loss += self._dissimilarity_penality(
                        theta, weights_[img_data][0], weights_[next_img_data][0]
                    )

        return total_loss

    def _create_covariance_matrix(
        self, matrix_1: np.ndarray, matrix_2: np.ndarray
    ) -> np.ndarray:
        """
        Function to calculate
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
            array of cross covariance matrix
        """
        return (matrix_1.T @ matrix_2) / matrix_1.shape[0]

    def _mean_center(self, data: np.ndarray) -> np.ndarray:
        """
        Function to demean data.

        Parmeteres
        ----------
        data: np.ndarray
            data to demean

        Returns
        -------
        np.ndarray: array
            demeaned data
        """
        return data - data.mean(axis=0)

    def _cross_cov_term(self, weight_beh, cov_mat, weight_img):
        return -weight_img.T @ (cov_mat @ weight_beh)

    def _regularization_term(self, weight, cov_mat, lambda_i):
        return 0.5 * lambda_i * (weight.T @ (cov_mat @ weight) - 1)

    def _dissimilarity_penality(self, theta_r, img_weight1, img_weight2):
        return theta_r * 0.5 * np.sum((img_weight1 - img_weight2) ** 2)
