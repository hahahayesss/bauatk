import flwr
import common
from typing import Dict, Optional, Tuple


def load_eval_fn(model):
    train_x, train_y, test_x, test_y = common.load_dataset()

    def _eval(
            server_round: int,
            parameters: flwr.common.NDArrays,
            config: Dict[str, flwr.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        model.set_weights(parameters)
        _loss, _acc = model.evaluate(test_x, test_y)
        print("-> Round: {}, Accuracy: {}".format(server_round, _acc))
        return _loss, {"accuracy": _acc}

    return _eval


if __name__ == '__main__':
    print("-> Compiling model")
    model = common.load_model()
    model.compile(
        "adam",
        "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("-> Starting server")
    flwr.server.start_server(
        config=flwr.server.ServerConfig(num_rounds=20),
        strategy=flwr.server.server.FedAvg(evaluate_fn=load_eval_fn(model))
    )
    print("--> Server started")
