import os

import torch
from torch.multiprocessing import Process, Queue
from transformers import AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = AutoModelForCausalLM.from_pretrained("gpt2")
tok_ids = torch.tensor(
    [[15205, 541, 305, 919, 278, 351, 12905, 2667, 15399, 714, 307, 281, 220]]
)


def fwd(model, tok_ids, queue):
    print("Starting process")
    print(f"{os.environ['TOKENIZERS_PARALLELISM']=}")
    print(f"{type(model)=}")
    print(f"{tok_ids=}")
    try:
        outs = model(tok_ids)
    except Exception as e:
        print(f"Error: {e}")
    print(f"{outs=}")
    queue.put(outs)


queue = Queue()
pr = Process(target=fwd, args=(model, tok_ids, queue))
pr.start()
pr.join()
outs = queue.get()
print(outs)
