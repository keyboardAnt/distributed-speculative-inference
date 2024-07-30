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
                target_latency=ConfigLatency(
                    model="lmsys/vicuna-13b-v1.3",
                    revision="refs/pr/5",
                    dataset="cnn_dailymail",
                    subset="2.0.0",
                ),
                drafter_latency=ConfigLatency(
                    model="double7/vicuna-68m", dataset="cnn_dailymail", subset="2.0.0"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="lmsys/vicuna-13b-v1.3",
                    revision="refs/pr/5",
                    dataset="danielkorat/alpaca",
                ),
                drafter_latency=ConfigLatency(
                    model="double7/vicuna-68m", dataset="danielkorat/alpaca"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="lmsys/vicuna-7b-v1.3",
                    revision="refs/pr/4",
                    dataset="cnn_dailymail",
                    subset="2.0.0",
                ),
                drafter_latency=ConfigLatency(
                    model="double7/vicuna-68m", dataset="cnn_dailymail", subset="2.0.0"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="lmsys/vicuna-7b-v1.3",
                    revision="refs/pr/4",
                    dataset="danielkorat/alpaca",
                ),
                drafter_latency=ConfigLatency(
                    model="double7/vicuna-68m", dataset="danielkorat/alpaca"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="bigcode/starcoder",
                    revision="refs/pr/112",
                    dataset="openai/openai_humaneval",
                ),
                drafter_latency=ConfigLatency(
                    model="bigcode/tiny_starcoder_py", dataset="openai/openai_humaneval"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="bigcode/starcoder", revision="refs/pr/112", dataset="mbpp"
                ),
                drafter_latency=ConfigLatency(
                    model="bigcode/tiny_starcoder_py", dataset="mbpp"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="microsoft/Phi-3-medium-128k-instruct",
                    dataset="openai/openai_humaneval",
                ),
                drafter_latency=ConfigLatency(
                    model="microsoft/Phi-3-mini-128k-instruct",
                    dataset="openai/openai_humaneval",
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="microsoft/Phi-3-medium-128k-instruct", dataset="mbpp"
                ),
                drafter_latency=ConfigLatency(
                    model="microsoft/Phi-3-mini-128k-instruct", dataset="mbpp"
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="microsoft/Phi-3-medium-128k-instruct",
                    dataset="cnn_dailymail",
                    subset="2.0.0",
                ),
                drafter_latency=ConfigLatency(
                    model="microsoft/Phi-3-mini-128k-instruct",
                    dataset="cnn_dailymail",
                    subset="2.0.0",
                ),
            ),
            ConfigTableRecord(
                target_latency=ConfigLatency(
                    model="microsoft/Phi-3-medium-128k-instruct",
                    dataset="danielkorat/alpaca",
                ),
                drafter_latency=ConfigLatency(
                    model="microsoft/Phi-3-mini-128k-instruct",
                    dataset="danielkorat/alpaca",
                ),
            ),
        ]
    )
