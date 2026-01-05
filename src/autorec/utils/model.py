import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_hidden_units: int,
        item_based: bool = True,
    ) -> None:
        super(AutoRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden_units = num_hidden_units
        self.item_based = item_based

        # For item-based AutoRec: input is item vector (num_users ratings)
        # For user-based AutoRec: input is user vector (num_items ratings)
        input_dim = num_users if item_based else num_items
        output_dim = num_users if item_based else num_items

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units), nn.Sigmoid()
        )

        self.decoder = nn.Sequential(nn.Linear(num_hidden_units, output_dim))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder(input_data)
        return self.decoder(encoder)