from pydantic import BaseModel, Field

class ConfigAcceptanteRate(BaseModel):
    """Includes all the parameters needed for measuring the acceptance rate 
    of a (target, draft, dataset) triplet.
    """
    seed: int = Field(42, 
                      title="The random seed for each experiment")
    num_ex: int = Field(5, 
                        title="The number of examples per dataset", ge=1)
    max_new_tokens: int = Field(256, 
                                title="The maximum number of new tokens to generate", 
                                ge=1)
    compiled_model: bool = Field(False,
                                 title="Whether to torch.compile() the model")
    do_sample_target: bool = Field(False,
                            title="whether enable sampling decoding strategies for the target model")
    do_sample_draft: bool = Field(False,
                            title="whether enable sampling decoding strategies for the draft model")
    temp_target: int = Field(1
                        title="temperature for the target model", 
                        ge=0)
    temp_draft: int = Field(1,
                        title="temperature for the draft model", 
                        ge=0)