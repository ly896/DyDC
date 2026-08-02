import torch
from torch import nn


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        raise NotImplementedError


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__()
        self.memory = memory
        self.message_dimension = message_dimension
        self.memory_dimension = memory_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) == 0:
            return

        assert (
            self.memory.get_last_update(unique_node_ids) <= timestamps
        ).all().item(), "Trying to update memory to time in the past."

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps
        updated_memory = self.memory_updater(unique_messages, memory)
        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) == 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_messages, updated_memory[unique_node_ids]
        )

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps
        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.RNNCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )


class GatedGRUMemoryUpdater(SequenceMemoryUpdater):
    """GRU memory update with a consistency gate for noisy interactions."""

    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )
        self.gate_net = nn.Sequential(
            nn.Linear(message_dimension + memory_dimension, memory_dimension // 2),
            nn.ReLU(),
            nn.Linear(memory_dimension // 2, 1),
            nn.Sigmoid(),
        )

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) == 0:
            return

        old_memory = self.memory.get_memory(unique_node_ids)
        candidate_memory = self.memory_updater(unique_messages, old_memory)
        gate_score = self.gate_net(torch.cat([unique_messages, old_memory], dim=1))
        updated_memory = gate_score * candidate_memory + (1 - gate_score) * old_memory

        self.memory.last_update[unique_node_ids] = timestamps
        self.memory.set_memory(unique_node_ids, updated_memory)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    if module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    if module_type == "gated_gru":
        return GatedGRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    raise ValueError(f"Memory updater type {module_type} is not supported.")
