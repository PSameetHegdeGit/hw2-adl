## CREDITS: With support from GPT-4o through Githuh Copilot

import abc

import torch
import math


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2 ** 10):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=1024
        )
        self.fc = torch.nn.Linear(d_latent, n_tokens)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Assume a tensor of shape (B, h, w) of integers
        print(f"tensor shape prior to embedding: {x.shape}\n")
        # TODO flatten tensor into a sequence
        B, h, w = x.shape
        x = x.flatten(1)

        # TODO generate the square sequence mask
        seq_len = x.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
        print("sequence length: ", seq_len)
        print("mask shape: ", mask.shape)

        x = self.embedding(x)
        print(f"tensor shape after embedding: {x.shape}\n")

        # TODO shift sequence by 1 position
        pad = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        x = torch.cat([pad, x[:, :-1, :]], dim=1)

        print(f"shifted tensor shape: {x.shape}\n")

        # TODO pass through the transformer
        x = self.transformer(x, mask)
        print("shape after transformer: ", x.shape)

        # TODO produce a probability over the next token
        x = self.fc(x)
        x = x.view(B, h, w, -1)

        return x, {}



    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        return 0
