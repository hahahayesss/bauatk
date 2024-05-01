import flwr


class FlwrClient(flwr.client.NumPyClient):
    def __init__(self, model, train_x, train_y, test_x, test_y):
        self.model = model
        self.x_train, self.y_train = train_x, train_y
        self.x_test, self.y_test = test_x, test_y

    def get_properties(self, config):
        raise Exception("Not implemented")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            32,
            3,
            validation_data=(self.x_test, self.y_test),
            verbose=0
        )

        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, verbose=0)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
