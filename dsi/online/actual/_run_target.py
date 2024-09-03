import logging

import torch
from transformers import AutoModelForCausalLM


class Target:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.pkv = None

        # batch, head, seq, dim/head

    def fix_pkv(self, start_id):
        self.pkv = tuple(
            (k[:, :, :start_id, :], v[:, :, :start_id, :]) for k, v in self.pkv
        )

    def verify(self, input_ids, start_id):
        """
        start_id: absolute index
        """
        if self.pkv is not None:
            current_pkv_len = self.pkv[0][0].shape[2]

            if start_id < current_pkv_len:
                self.fix_pkv(start_id)
            else:
                start_id = current_pkv_len

        input_ids_check = input_ids[:, start_id:]

        outputs = self.model(input_ids_check, past_key_values=self.pkv)[0]

        logits = outputs.logits

        self.pkv = outputs.past_key_values

        target_pred = logits.argmax(dim=-1)
        target_pred = target_pred[:, :-1]
        input_ids_check_continue = input_ids_check[:, 1:]

        logging.error(f"{input_ids_check_continue=}, {target_pred=}")

        draft_target_agree = input_ids_check != target_pred

        logging.error(f"{draft_target_agree=}")

        first_mistake_index_list = draft_target_agree.nonzero()

        if len(first_mistake_index_list) == 0:
            # no mistakes
            mistake_index = None
            correct_token = None
        else:
            logging.error(f"{first_mistake_index_list=}")
            mistake_index = first_mistake_index_list[0][1]
            correct_token = target_pred[0, mistake_index]
        logging.error(f"{correct_token=}, {mistake_index=}")

        return correct_token, mistake_index
