from torch import nn


class MessageFunction(nn.Module):
    def compute_message(self, raw_messages):
        raise NotImplementedError


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension, message_dimension):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        )

    def compute_message(self, raw_messages):
        return self.mlp(raw_messages)


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages):
        return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    if module_type == "identity":
        return IdentityMessageFunction()
    raise ValueError(f"Message function type {module_type} is not supported.")
