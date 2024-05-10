from typing import Any
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle


class MLP:
    def __init__(
        self,
        X: Any,
        y: Any,
    ) -> None:
        self.mlp_model = None
        self.y_test = None
        self.y_train = None
        self.x_test_scaled = None
        self.x_train_scaled = None
        self.y_hat_test = None
        self.y_hat_train = None
        self.X = X
        self.y = y
        self.number_of_features = len(self.X.columns)
        self.max_iterations = 1000
        self.mlp_structure = (
            self.number_of_features,
            self.number_of_features * 4,
            self.number_of_features * 8,
            self.number_of_features * 8,
            self.number_of_features * 4,
            self.number_of_features,
        )

    def get_mlp_model(self) -> Any:
        mlp = MLPRegressor(
            hidden_layer_sizes=self.mlp_structure,
            max_iter=self.max_iterations,
            tol=1e-7,
            n_iter_no_change=5,
            activation="relu",
            solver="adam",
            random_state=13,
        )
        return mlp

    def process_dataset(self, X, y) -> Any:
        features_array_x = X.to_numpy()
        target_vector_y = y.to_numpy()
        target_vector_y = target_vector_y.reshape(-1, 1).ravel()
        x_train, x_test, y_train, y_test = train_test_split(
            features_array_x, target_vector_y, test_size=0.2, random_state=42
        )
        self.input_scaler = StandardScaler()
        self.x_train_scaled = self.input_scaler.fit_transform(x_train)
        with open('models/scalar.pkl', 'wb') as file:
            pickle.dump(self.input_scaler, file)
        self.x_test_scaled = self.input_scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test

    def train(self) -> Any:
        self.mlp_model = self.get_mlp_model()
        self.process_dataset(self.X, self.y)
        self.mlp_model.fit(self.x_train_scaled, self.y_train)
        with open('models/mlp_regressor_model_' + list(self.y.columns)[0] + '.pkl', 'wb') as file:
            pickle.dump(self.mlp_model, file)
        return self.mlp_model

    def get_performance_metrics(self):
        self.y_hat_train = self.mlp_model.predict(self.x_train_scaled)
        self.y_hat_test = self.mlp_model.predict(self.x_test_scaled)
        train_mse = mean_squared_error(self.y_train, self.y_hat_train)
        test_mse = mean_squared_error(self.y_test, self.y_hat_test)
        return train_mse, test_mse

    def predict(self, test_row, variables):
        final_output = []
        test_row = test_row.to_numpy()
        with open('models/scalar.pkl', 'rb') as file:
            input_scaler = pickle.load(file)
        test_row_scaled = input_scaler.transform(test_row)
        for variable in variables:
            prediction_model_name = 'models/mlp_regressor_model_' + variable + '.pkl'
            with open(prediction_model_name, 'rb') as file:
                prediction_model = pickle.load(file)
                final_output.append(prediction_model.predict(test_row_scaled))
        return final_output






