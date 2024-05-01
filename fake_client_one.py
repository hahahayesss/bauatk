import flwr
import common
from client import FlwrClient

if __name__ == '__main__':
    print("-> Compiling model")
    model = common.load_model()
    model.compile(
        "adam",
        "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    x_train, y_train, x_test, y_test = common.load_dataset()
    x_train, y_train = common.slice_data(
        [4000, 10, 4000, 10, 4000, 10, 4000, 10, 4000, 10],
        x_train, y_train
    )

    print("-> Starting client")
    flwr.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlwrClient(
            model,
            x_train, y_train, x_test, y_test
        )
    )
