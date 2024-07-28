from typing import Any

from pydantic import BaseModel, Field

from dsi.configs.experiment.latency import ConfigLatency
from dsi.types.exception import DatasetMismatchError


class ConfigTableRecord(BaseModel):
    target_latency: ConfigLatency
    drafter_latency: ConfigLatency

    def model_post_init(self, __context: Any) -> None:
        if self.target_latency.dataset != self.drafter_latency.dataset:
            raise DatasetMismatchError(
                target_dataset=self.target_latency.dataset,
                drafter_dataset=self.drafter_latency.dataset,
            )
        return super().model_post_init(__context)


class ConfigTable(BaseModel):
    records: list[ConfigTableRecord] = Field(
        default=[
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Vicuna-13B", dataset="CNN-DM"),
                drafter_latency=ConfigLatency(model="Vicuna-68M", dataset="CNN-DM"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Vicuna-13B", dataset="Alpaca"),
                drafter_latency=ConfigLatency(model="Vicuna-68M", dataset="Alpaca"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Vicuna-7B", dataset="CNN-DM"),
                drafter_latency=ConfigLatency(model="Vicuna-68M", dataset="CNN-DM"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Vicuna-7B", dataset="Alpaca"),
                drafter_latency=ConfigLatency(model="Vicuna-68M", dataset="Alpaca"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="Starcoder-15B", dataset="HumanEval"
                ),
                drafter_latency=ConfigLatency(
                    model="Starcoder-168M", dataset="HumanEval"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Starcoder-15B", dataset="MBPP"),
                drafter_latency=ConfigLatency(model="Starcoder-168M", dataset="MBPP"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Phi3-14B", dataset="HumanEval"),
                drafter_latency=ConfigLatency(model="Phi3-4B", dataset="HumanEval"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Phi3-14B", dataset="MBPP"),
                drafter_latency=ConfigLatency(model="Phi3-4B", dataset="MBPP"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Phi3-14B", dataset="CNN-DM"),
                drafter_latency=ConfigLatency(model="Phi3-4B", dataset="CNN-DM"),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(model="Phi3-14B", dataset="Alpaca"),
                drafter_latency=ConfigLatency(model="Phi3-4B", dataset="Alpaca"),
            ),
        ]
    )
