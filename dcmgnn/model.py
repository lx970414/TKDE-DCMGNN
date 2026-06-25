from __future__ import annotations

import itertools

import torch
from torch import nn
import torch.nn.functional as F

from .data import BehaviorDataset


class DCMGNN(nn.Module):
    """Dual-channel multiplex GNN aligned with the TKDE DCMGNN design.

    The explicit channel learns BBP-aware node embeddings. The implicit channel
    learns relation-specific and relation-chain embeddings ordered by the target
    behavior chain, for example View -> Cart -> Buy on Tmall and Retail.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        behaviors: tuple[str, ...],
        target_behavior: str,
        relation_order: tuple[str, ...],
        num_bbps: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        behavior_layers: tuple[int, ...] | None = None,
        fusion_mode: str = "static",
        prior_behavior: str | None = None,
        prior_alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.behaviors = behaviors
        self.target_behavior = target_behavior
        self.relation_order = relation_order
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        if fusion_mode not in {"static", "dynamic"}:
            raise ValueError("fusion_mode must be 'static' or 'dynamic'.")
        self.fusion_mode = fusion_mode
        if prior_behavior is not None and prior_behavior not in behaviors:
            raise ValueError("prior_behavior must be one of the configured behaviors.")
        self.prior_behavior = prior_behavior
        self.prior_alpha = prior_alpha
        if behavior_layers is None:
            behavior_layers = tuple(num_layers for _ in behaviors)
        if len(behavior_layers) != len(behaviors):
            raise ValueError("behavior_layers must have the same length as behaviors.")
        self.behavior_layers = dict(zip(behaviors, behavior_layers))
        self.dropout = nn.Dropout(dropout)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.bbp_weight = nn.Parameter(torch.zeros(max(num_bbps, 1)))
        self.global_bbp_weight = nn.Parameter(torch.zeros(max(num_bbps, 1)))
        self.relation_weight = nn.Parameter(torch.zeros(len(behaviors)))
        self.channel_weight = nn.Parameter(torch.zeros(4))
        self.channel_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embedding_dim, 4),
        )
        self.contrast_weight_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embedding_dim, 2),
        )
        self.final_log_scale = nn.Parameter(torch.tensor(0.0))

        chain_specs = self._make_chain_specs(relation_order, target_behavior)
        self.chain_specs = chain_specs
        self.chain_weight = nn.Parameter(torch.zeros(len(chain_specs)))
        self.chain_transforms = nn.ModuleDict()
        self.cascade_transforms = nn.ModuleDict()
        for chain in chain_specs:
            for src, dst in zip(chain[:-1], chain[1:]):
                key = f"{src}__to__{dst}"
                if key not in self.chain_transforms:
                    self.chain_transforms[key] = nn.Linear(embedding_dim, embedding_dim, bias=False)
        for src, dst in zip(relation_order[:-1], relation_order[1:]):
            key = f"{src}__to__{dst}"
            self.cascade_transforms[key] = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for module in self.chain_transforms.values():
            nn.init.xavier_uniform_(module.weight)
        for module in self.cascade_transforms.values():
            nn.init.xavier_uniform_(module.weight)
        for module in self.channel_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.contrast_weight_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def contrast_weights(
        self,
        final: torch.Tensor,
        chain: torch.Tensor,
        target_relation: torch.Tensor,
        users: torch.Tensor,
        scale: float = 0.5,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                F.normalize(final[users], dim=-1),
                F.normalize(chain[users], dim=-1),
                F.normalize(target_relation[users], dim=-1),
            ),
            dim=-1,
        )
        logits = self.contrast_weight_net(features)
        return 1.0 + scale * torch.tanh(logits)

    @classmethod
    def from_dataset(
        cls,
        dataset: BehaviorDataset,
        embedding_dim: int = 64,
        num_layers: int = 3,
        behavior_layers: tuple[int, ...] | None = None,
        fusion_mode: str = "static",
        prior_behavior: str | None = None,
        prior_alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> "DCMGNN":
        return cls(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            behaviors=dataset.config.behaviors,
            target_behavior=dataset.config.target_behavior,
            relation_order=dataset.config.relation_order,
            num_bbps=len(dataset.bbp_adjs),
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            behavior_layers=behavior_layers,
            fusion_mode=fusion_mode,
            prior_behavior=prior_behavior,
            prior_alpha=prior_alpha,
            dropout=dropout,
        )

    def forward(self, dataset: BehaviorDataset) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        initial = self.all_embeddings()
        explicit = self._explicit_behavior_pattern_embeddings(initial, dataset)
        relation_embeddings = self._relation_embeddings(initial, dataset)
        relation_stack = torch.stack([relation_embeddings[name] for name in self.behaviors], dim=0)
        relation_weights = torch.softmax(self.relation_weight, dim=0).view(-1, 1, 1)
        relation_sum = len(self.behaviors) * (relation_weights * relation_stack).sum(dim=0)
        cascade_embeddings = self._cascade_embeddings(initial, dataset)
        cascade_target = cascade_embeddings[self.target_behavior]
        chain = self._relation_chain_embeddings(relation_embeddings)
        channels = torch.stack((explicit, relation_sum, cascade_target, chain), dim=0)
        if self.fusion_mode == "dynamic":
            gate_input = torch.cat(
                (
                    F.normalize(explicit, dim=-1),
                    F.normalize(relation_sum, dim=-1),
                    F.normalize(cascade_target, dim=-1),
                    F.normalize(chain, dim=-1),
                ),
                dim=-1,
            )
            dynamic_logits = self.channel_gate(gate_input)
            weights = torch.softmax(dynamic_logits + self.channel_weight.view(1, 4), dim=-1)
            final = (weights.T.unsqueeze(-1) * channels).sum(dim=0)
            channel_weights = weights
        else:
            weights = torch.softmax(self.channel_weight, dim=0).view(4, 1, 1)
            final = (weights * channels).sum(dim=0)
            channel_weights = torch.softmax(self.channel_weight, dim=0).view(1, 4).expand(
                channels.size(1), -1
            )
        if self.prior_behavior is not None:
            prior = relation_embeddings[self.prior_behavior]
            final = self.prior_alpha * final + (1.0 - self.prior_alpha) * prior
        final = self.dropout(torch.exp(self.final_log_scale).clamp(max=10.0) * final)

        return {
            "final": final,
            "explicit": explicit,
            "relation_sum": relation_sum,
            "cascade_sum": cascade_target,
            "chain": chain,
            "channel_weights": channel_weights,
            "relations": relation_embeddings,
            "cascade_relations": cascade_embeddings,
        }

    def all_embeddings(self) -> torch.Tensor:
        return torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)

    def split_user_item(self, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return node_embeddings[: self.num_users], node_embeddings[self.num_users :]

    def _explicit_behavior_pattern_embeddings(
        self,
        initial: torch.Tensor,
        dataset: BehaviorDataset,
    ) -> torch.Tensor:
        if not dataset.bbp_adjs:
            local = initial
            global_pattern = initial
        else:
            weights = torch.softmax(self.bbp_weight[: len(dataset.bbp_adjs)], dim=0)
            h = initial
            outputs = []
            for _ in range(self.num_layers):
                messages = []
                for weight, adj in zip(weights, dataset.bbp_adjs):
                    messages.append(weight * torch.sparse.mm(adj, h))
                h = torch.stack(messages, dim=0).sum(dim=0)
                outputs.append(h)
            local = torch.stack(outputs, dim=0).mean(dim=0)

            pattern_weights = torch.softmax(
                self.global_bbp_weight[: dataset.pattern_features.size(1)], dim=0
            )
            q = dataset.pattern_features * pattern_weights.view(1, -1)
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            h_global = initial
            global_outputs = []
            for _ in range(self.num_layers):
                h_global = q @ (q.T @ h_global)
                h_global = F.normalize(h_global, dim=-1)
                global_outputs.append(h_global)
            global_pattern = torch.stack(global_outputs, dim=0).mean(dim=0)
        return F.normalize((local + global_pattern) * 0.5, dim=-1)

    def _relation_embeddings(
        self,
        initial: torch.Tensor,
        dataset: BehaviorDataset,
    ) -> dict[str, torch.Tensor]:
        relation_embeddings: dict[str, torch.Tensor] = {}
        for behavior in self.behaviors:
            adj = dataset.relation_adjs[behavior]
            h = initial
            outputs = [h]
            for _ in range(self.behavior_layers[behavior]):
                h = torch.sparse.mm(adj, h)
                outputs.append(h)
            relation_embeddings[behavior] = F.normalize(torch.stack(outputs, dim=0).sum(dim=0), dim=-1)
        return relation_embeddings

    def _cascade_embeddings(
        self,
        initial: torch.Tensor,
        dataset: BehaviorDataset,
    ) -> dict[str, torch.Tensor]:
        cascade_embeddings: dict[str, torch.Tensor] = {}
        h0 = initial
        previous_behavior: str | None = None
        for behavior in self.relation_order:
            if previous_behavior is not None:
                h0 = self.cascade_transforms[f"{previous_behavior}__to__{behavior}"](h0)
                h0 = F.leaky_relu(h0, negative_slope=0.2)

            h = h0
            outputs = [h]
            for _ in range(self.behavior_layers[behavior]):
                h = torch.sparse.mm(dataset.relation_adjs[behavior], h)
                outputs.append(h)
            h0 = torch.stack(outputs, dim=0).sum(dim=0)
            cascade_embeddings[behavior] = F.normalize(h0, dim=-1)
            previous_behavior = behavior
        return cascade_embeddings

    def _relation_chain_embeddings(
        self,
        relation_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        chain_outputs: list[torch.Tensor] = []
        for chain in self.chain_specs:
            h = relation_embeddings[chain[0]]
            for src, dst in zip(chain[:-1], chain[1:]):
                h = self.chain_transforms[f"{src}__to__{dst}"](h)
                h = F.leaky_relu(h, negative_slope=0.2)
            chain_outputs.append(h)
        if not chain_outputs:
            return relation_embeddings[self.target_behavior]
        chain_stack = torch.stack(chain_outputs, dim=0)
        weights = torch.softmax(self.chain_weight[: len(chain_outputs)], dim=0).view(-1, 1, 1)
        return F.normalize(len(chain_outputs) * (weights * chain_stack).sum(dim=0), dim=-1)

    @staticmethod
    def _make_chain_specs(
        relation_order: tuple[str, ...],
        target_behavior: str,
    ) -> list[tuple[str, ...]]:
        chains: list[tuple[str, ...]] = []
        before_target = [name for name in relation_order if name != target_behavior]
        for size in range(1, len(before_target) + 1):
            for combo in itertools.combinations(before_target, size):
                chains.append(tuple(combo) + (target_behavior,))
        if (target_behavior,) not in chains:
            chains.append((target_behavior,))
        return chains
