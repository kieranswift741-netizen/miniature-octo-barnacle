"""
LSTM model for predicting quantum battery density matrix evolution.

Architecture:
  Input:   [batch, window_size, feature_dim] — past ρ snapshots
  LSTM:    2-3 stacked LSTM layers
  Dense:   Dense → ReLU → Dropout → Dense → output
  Output:  [batch, output_dim] — predicted ρ(t+Δt) real + imag parts
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class QuantumBatteryLSTM(nn.Module):
    """
    Stacked LSTM for predicting the next density matrix from a
    sliding window of past density matrix snapshots.

    The LSTM hidden state naturally encodes the environmental
    memory (non-Markovian correlation function).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dense_units: int = 16,
        output_dim: int = 8,
        dropout: float = 0.2,
        return_hidden: bool = False,
        cond_dim: int = 0,  # Conditional parameter dimension
    ):
        """
        Args:
            input_dim: Feature dimension of each time step (2 * d² for complex ρ).
            hidden_dims: List of hidden sizes for each LSTM layer.
            dense_units: Size of intermediate Dense layer.
            output_dim: Output dimension (2 * d² for predicted ρ).
            dropout: Dropout rate between LSTM layers.
            return_hidden: If True, forward() also returns final hidden states.
            cond_dim: Dimension of conditional parameters (Ω, γ, τ_m, Ω/γ).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.return_hidden = return_hidden
        self.cond_dim = cond_dim

        # Conditional parameter encoder
        # Encode to input_dim so we can concatenate with each time step
        if cond_dim > 0:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),  # Match input_dim for concatenation
            )
            # LSTM input will be doubled after concatenation
            lstm_input_dim = input_dim * 2
        else:
            lstm_input_dim = input_dim

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dims[0],
            num_layers=len(hidden_dims),
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0.0,
        )

        # Fully connected head
        # No final activation — model outputs raw values for MSE loss.
        # When using feature standardization, targets are ~N(0,1) and
        # Tanh would squash them. Physics constraints handle bounding.
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[0], dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_dim).
            h0: Optional initial hidden state.
            cond: Optional conditional parameters (batch, cond_dim).

        Returns:
            If return_hidden=False: output tensor (batch, output_dim).
            If return_hidden=True: (output, (h_n, c_n)).
        """
        # Handle conditional parameters
        if self.cond_dim > 0:
            if cond is not None:
                # Encode condition to (batch, input_dim)
                cond_feat = self.cond_encoder(cond)
            else:
                # If no condition provided, use zero vector
                batch_size = x.size(0)
                cond_feat = torch.zeros(batch_size, self.input_dim, device=x.device)
            
            # Expand to (batch, window_size, input_dim)
            cond_feat = cond_feat.unsqueeze(1).expand(-1, x.size(1), -1)
            # Concatenate: (batch, window_size, input_dim*2)
            x = torch.cat([x, cond_feat], dim=-1)

        lstm_out, (h_n, c_n) = self.lstm(x, h0)

        # Use the last time step's output
        last_out = lstm_out[:, -1, :]

        output = self.fc(last_out)

        if self.return_hidden:
            return output, (h_n, c_n)
        return output

    def predict_sequence(
        self,
        initial_window: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Autoregressive prediction: roll the window forward to predict
        multiple future steps.

        Args:
            initial_window: Shape (1, window_size, input_dim) — starting window.
            n_steps: Number of future steps to predict.

        Returns:
            Predictions of shape (n_steps, output_dim).
        """
        self.eval()
        window = initial_window.clone()
        predictions = []

        with torch.no_grad():
            for _ in range(n_steps):
                pred = self(window.unsqueeze(0)).squeeze(0)  # (output_dim,)
                predictions.append(pred)

                # Roll window: drop first step, append prediction
                window = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)

        return torch.stack(predictions)

    def get_forget_gate_weights(self) -> torch.Tensor:
        """
        Extract forget gate weights from the first LSTM layer for analysis.
        Higher forget gate activation → shorter memory.
        """
        # LSTM weight layout: [W_ii|W_if|W_ig|W_io] stacked for each layer
        weight_ih = self.lstm.weight_ih_l0  # (4 * hidden, input)
        hidden_size = self.hidden_dims[0]
        # Forget gate weights are the second quarter
        w_if = weight_ih[hidden_size : 2 * hidden_size]
        return w_if.detach()


def build_model(
    input_dim: int,
    output_dim: int,
    lstm_layers: List[int] = None,
    dense_units: int = 16,
    dropout: float = 0.2,
    return_hidden: bool = False,
    cond_dim: int = 0,
) -> QuantumBatteryLSTM:
    """
    Factory function to build the LSTM model from config parameters.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension (2 * d² for complex ρ).
        lstm_layers: Hidden sizes per LSTM layer (default: [64, 32]).
        dense_units: Dense layer units.
        dropout: Dropout rate.
        return_hidden: Whether to return hidden states.
        cond_dim: Conditional parameter dimension (Ω, γ, τ_m, Ω/γ).

    Returns:
        Configured QuantumBatteryLSTM model.
    """
    if lstm_layers is None:
        lstm_layers = [64, 32]

    return QuantumBatteryLSTM(
        input_dim=input_dim,
        hidden_dims=lstm_layers,
        dense_units=dense_units,
        output_dim=output_dim,
        dropout=dropout,
        return_hidden=return_hidden,
        cond_dim=cond_dim,
    )


if __name__ == "__main__":
    # Quick test
    model = QuantumBatteryLSTM(
        input_dim=8,    # N=1 TLS: 2 * 2² = 8
        hidden_dims=[64, 32],
        dense_units=16,
        output_dim=8,
    )

    x = torch.randn(4, 20, 8)  # batch=4, window=20, features=8
    out = model(x)
    print(f"Model: {model}")
    print(f"Input shape: {x.shape} → Output shape: {out.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
