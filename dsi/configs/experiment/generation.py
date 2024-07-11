from pydantic import Field

from dsi.configs.experiment.base import _ConfigExperiment
from dsi.types.exception import (
    InvalidGenConfigError,
)

class ConfigGen(_ConfigExperiment):
    """Includes all the parameters needed for measuring the latencies
    of a (target, draft, dataset) triplet.
    """

    do_sample: bool = Field(False, title="Whether to use sampling during generation")
    temperature: float = Field(1.0, title="The temperature value for generation", ge=0.0)
    top_p: float = Field(1.0, title="The top-p value for generation", ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        """
        Verify configuration validity.
        """
        if self.do_sample and self.temperature == 0:
            raise InvalidGenConfigError("temperature must be different than 0 when do_sample is True.")
        elif not self.do_sample:
            if self.temperature is not None and self.temperature != 1.0:
                raise InvalidGenConfigError("temperature must be 1.0 when do_sample is False.")
            if self.top_p is not None and self.top_p != 1.0:
                raise InvalidGenConfigError("top_p must be 1.0 when do_sample is False.")