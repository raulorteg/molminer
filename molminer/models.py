import torch
import torch.nn as nn
import torch.nn.functional as F


class FragmentStarter(nn.Module):
    """
    MLP classifier used to select an initial fragment from the vocabulary,
    conditioned on a property vector.
    """

    def __init__(
        self,
        d_model_in: int,
        d_model_out: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        """
        Parameters
        ----------
        d_model_in : int
            Input feature size (e.g., property vector).
        d_model_out : int
            Number of output classes (fragment vocabulary size).
        d_ff : int
            Hidden layer width.
        dropout : float
            Dropout probability.
        num_layers : int
            Number of hidden layers (excluding input/output).
        """
        super(FragmentStarter, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(d_model_in, d_ff))
        self.batch_norms.append(nn.BatchNorm1d(d_ff))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(d_ff, d_ff))
            self.batch_norms.append(nn.BatchNorm1d(d_ff))

        # Output layer
        self.output_layer = nn.Linear(d_ff, d_model_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d_model_in).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, d_model_out).
        """
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.dropout(F.gelu(layer(x)))
            x = batch_norm(x)

        logits = self.output_layer(x)  # raw logits
        return logits


class MoleculeTransformer(nn.Module):
    """
    Geometry-aware autoregressive transformer for fragment prediction.
    Takes current molecular state and predicts the next attachment.
    """

    def __init__(
        self,
        fragment_vocab_size: int,
        attachment_vocab_size: int,
        anchor_vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        num_properties: int,
        dropout: float,
        d_anchor: int,
        context_hidden_dim: int,
        context_n_layers: int,
        geom_requires_grad: bool,
        geom_init_value: float,
    ) -> None:
        """
        Parameters
        ----------
        fragment_vocab_size, attachment_vocab_size, anchor_vocab_size : int
            Vocabulary sizes for fragments, attachments, and anchors.
        d_model, d_ff : int
            Hidden and feedforward dimensions.
        n_heads : int
            Number of self-attention heads.
        num_layers : int
            Number of decoder layers.
        num_properties : int
            Size of the property vector.
        dropout : float
            Dropout rate.
        d_anchor : int
            Dimension of anchor embeddings.
        context_hidden_dim, context_n_layers : int
            Configuration for the FFNN that contextualizes fragments.
        geom_requires_grad : bool
            Whether the geometry attention weight is learnable.
        geom_init_value : float
            Initial value for geometry attention weight.
        """
        super(MoleculeTransformer, self).__init__()

        self.d_model = d_model

        # embeddings for fragments and anchors
        self.atom_embedding = nn.Embedding(fragment_vocab_size, d_model, padding_idx=0)
        self.anchor_embedding = nn.Embedding(anchor_vocab_size, d_anchor, padding_idx=0)

        # transformer layers
        self.layers = nn.ModuleList(
            [
                MoleculeDecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Note: this network is used to include into the fragments embeddings their local environment (saturations)
        # the objective is that it modifies the fragment representation with the context of
        # the local environment (how saturated it is)
        self.contextualise_fragments = FFNN(
            input_dim=d_model + 3,
            hidden_dim=context_hidden_dim,
            output_dim=d_model,
            n_layers=context_n_layers,
            dropout_rate=dropout,
        )

        # Note: This function takes the embedding of the focal fragment after all transformer layers, and given the embedding of the
        # type of attachment (e.g [c,c]) it outputs the logits for the probility distribution of the vocabulary of next_tokens
        self.output_projection = FFNN(
            input_dim=d_model + d_anchor + num_properties,
            hidden_dim=2 * (d_model + d_anchor),
            output_dim=attachment_vocab_size,
            n_layers=2,
            dropout_rate=dropout,
        )

        # this scalar is used to weight the effect of the geometry in the attention mechanism. If set to zero
        # then geometry is removed from the model.
        self.geometry_weight = nn.Parameter(
            torch.tensor(geom_init_value), requires_grad=geom_requires_grad
        )

    def forward(
        self,
        atom_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        focal_attachment_order: torch.Tensor,
        nodes_saturation: torch.Tensor,
        focal_attachment_type: torch.Tensor,
        properties: torch.Tensor,
        attn_mask_readout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Parameters
        ----------
        atom_ids : torch.Tensor
            Fragment token indices (B, T).
        attn_mask : torch.Tensor
            Geometry mask (B, T, T).
        focal_attachment_order : torch.Tensor
            Indices pointing to the focal node in each sample (B, 1).
        nodes_saturation : torch.Tensor
            Saturation values appended to embeddings (B, T, 3).
        focal_attachment_type : torch.Tensor
            Anchor type tokens (B, 1).
        properties : torch.Tensor
            Property vectors (B, 1, num_properties).
        attn_mask_readout : torch.Tensor
            Additional readout mask (B, 1, T).

        Returns
        -------
        torch.Tensor
            Logits over the attachment vocabulary (B, vocab_size).
        """

        # get the embeddings of the fragments
        x = self.atom_embedding(atom_ids)

        # add the local envoronment of the fragments into the embeddings
        x = self.contextualise_fragments(torch.cat((x, nodes_saturation), dim=-1))

        anchor_emb = self.anchor_embedding(focal_attachment_type).squeeze(1)

        attn_mask = self.geometry_weight * attn_mask

        # Pass through the decoder layers with the pairwise distances as the custom mask
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Extract the specific elements from `x` first using focal_attachment_order
        # Note: query will have shape (batch_size, 1, feature_dim)
        query = x[torch.arange(x.size(0)), focal_attachment_order.squeeze(1)].unsqueeze(
            1
        )

        # The result will be (batch_size, 1, max_seq_len)
        attn_scores = (
            torch.bmm(query, x.transpose(1, 2)) / (self.d_model**0.5)
            + self.geometry_weight * attn_mask_readout
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, x).squeeze(1)

        # Concatenate the focal embedding with the properties and anchor embedding
        # to project over the attachment vocabulary
        x_selected = torch.cat((attn_output, anchor_emb, properties[:, 0, :]), dim=-1)

        # Apply the projection only to the selected elements
        logits_next = self.output_projection(x_selected.unsqueeze(1)).squeeze(
            1
        )  # Shape: (batch_size, num_classes)

        return logits_next


class MoleculeDecoderBlock(nn.Module):
    """
    Single transformer decoder block with attention and feed-forward layers.
    Supports geometry-based attention masking.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(MoleculeDecoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings (B, T, d_model).
        attn_mask : torch.Tensor
            Geometry-aware attention mask (B * n_heads, T, T).

        Returns
        -------
        torch.Tensor
            Transformed sequence (B, T, d_model).
        """

        # Repeat the distance mask over the number of attention heads (batch_size * n_heads, seq_len, seq_len)
        attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        # Self-attention with the custom pairwise distance mask
        attn_output, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,  # custom mask based on pairwise distances
            key_padding_mask=None,
        )

        # Residual connection and layer norm
        x = self.layer_norm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x


class FFNN(nn.Module):
    """
    Fully connected feed-forward network with optional depth and dropout.
    Used for fragment contextualization and output projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout_rate: float,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Feature size after concatenation.
        hidden_dim : int
            Size of hidden layers.
        output_dim : int
            Output feature size.
        n_layers : int
            Number of hidden layers.
        dropout_rate : float
            Dropout probability.
        """
        super(FFNN, self).__init__()

        layers = []

        # Initial input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine layers
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, output_dim).
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, x.shape[-1])  # Flatten (batch * seq_len, features)

        x = self.model(x)

        # Reshape back to (batch, seq_len, output_dim)
        return x.view(batch_size, seq_len, -1)
