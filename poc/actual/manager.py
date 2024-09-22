import torch
from poc.actual.pubsub import PubSub
from poc.actual.event import Preemption, Request, Response


import asyncio
from typing import Dict, Tuple
from uuid import UUID


class Manager:
    """
    Manages the overall system, handling requests, responses, and preemptions.

    Assumptions:
    - There is only one Manager instance in the system.
    - The Manager has exclusive access to modify the request queues.

    Guarantees:
    - Preemptions are broadcasted to all workers via PubSub.
    - Requests and responses older than the last preemption will be dropped.

    Attributes:
        draft_queue (asyncio.Queue): Queue for draft requests.
        verify_queue (asyncio.Queue): Queue for verification requests.
        response_queue (asyncio.Queue): Queue for responses from workers.
        pubsub (PubSub): PubSub system for broadcasting preemptions.
        tok_ids (torch.Tensor): Token IDs of the prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        vocab_size (int): Dimension of the output scores.
        timestamp_preemption (float): Timestamp of the last preemption.
    """

    def __init__(
        self,
        draft_queue: asyncio.Queue[Request],
        verify_queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        pubsub: PubSub,
        tok_ids: torch.Tensor,
        max_new_tokens: int,
        vocab_size: int,
        lookahead: int,
    ):
        print(f"{self.__class__.__name__}: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.response_queue = response_queue
        self.seq_len = tok_ids.shape[1] + max_new_tokens
        self.tok_ids = torch.full(
            (1, self.seq_len),
            -1,
            dtype=torch.int64,
        )
        self.tok_ids[:, : tok_ids.shape[1]] = tok_ids
        self.lookahead = lookahead
        # Initialize with -1s
        self.draft_scores = torch.full(
            (1, self.seq_len, vocab_size),
            -1,
            dtype=torch.float,
        )
        self.draft_tok_ids = torch.full(
            (
                1,
                self.seq_len,
            ),
            -1,
            dtype=torch.int64,
        )
        self.id_to_mask: Dict[UUID, torch.Tensor] = {}
        self.requested_verify = torch.full_like(
            self.draft_tok_ids, False, dtype=torch.bool
        )
        self.requested_draft = self.requested_verify.clone()
        self.pubsub = pubsub
        print(f"{self.__class__.__name__}: Initialized with PubSub")

    async def _send(self, request: Request, queue: asyncio.Queue[Request]) -> None:
        self.id_to_mask[request.id] = request.get_mask(
            seq_len=self.seq_len, is_draft=queue == self.draft_queue
        )
        requested = (
            self.requested_verify
            if queue == self.verify_queue
            else self.requested_draft
        )
        if requested[0, self.id_to_mask[request.id]].all():
            print(
                f"{self.__class__.__name__}: Won't send {('verify' if queue == self.verify_queue else 'draft')} request {request.id} because it covers already requested positions."
            )
            return
        requested[0, self.id_to_mask[request.id]] = True
        print(
            f"{self.__class__.__name__}: Enqueuing request {request.id} to {'draft' if queue == self.draft_queue else 'verify'} queue"
        )
        await queue.put(request)
        print(
            f"{self.__class__.__name__}: Sent {('verify' if queue == self.verify_queue else 'draft')} request with n={request.n} and tok_ids:\n{self.get_tok_ids_with_drafts()}"
        )

    def _reset(self) -> None:
        print(
            f"{self.__class__.__name__}: Resetting draft_scores, draft_tok_ids, and id_to_mask"
        )
        self.draft_scores.fill_(-1)
        self.draft_tok_ids.fill_(-1)
        self.id_to_mask.clear()
        self.requested_verify.fill_(False)
        self.requested_draft.fill_(False)

    async def preempt_all(self) -> None:
        """
        Broadcasts a preemption message to all workers and clears the request queues.

        Assumptions:
        - This method has exclusive access to modify the timestamp_preemption.

        Guarantees:
        - All workers will be notified of the preemption.
        - All request queues will be emptied.
        """
        print(f"{self.__class__.__name__}: Preempting all workers")
        # Send preempt message to workers
        print(f"{self.__class__.__name__}: Sending preempt message to workers")
        await self.pubsub.publish(Preemption.create())
        print(f"{self.__class__.__name__}: Preempt message sent to workers")
        # # Clear the queues
        # print(f"{self.__class__.__name__}: Clearing queues")
        # await self._empty_queue(self.draft_queue)
        # await self._empty_queue(self.verify_queue)
        # print(f"{self.__class__.__name__}: Queues cleared")

    async def run(self) -> None:
        print(f"{self.__class__.__name__}: Starting run")
        print(
            f"{self.__class__.__name__}: prompt's tok_ids.shape: {self.tok_ids.shape}"
        )
        print(f"{self.__class__.__name__}: prompt's tok_ids:\n{self.tok_ids}")
        to_verify_semaphore: int = self.verify_queue.maxsize
        print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
        to_draft: bool = True
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(
                f"{self.__class__.__name__}: number of empty tok_ids: {(self.tok_ids == -1).sum()}"
            )
            print(f"{self.__class__.__name__}: {self.tok_ids=}")
            any_rejected: bool = False
            if to_verify_semaphore > 0:
                await self.send_reqeust_verify()
                to_verify_semaphore -= 1
                print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
            if to_draft:
                await self.send_request_draft()
            to_draft = False
            while (self.tok_ids == -1).any():  # On dropping
                print(f"{self.__class__.__name__}: Waiting for response")
                response: Response = await self.response_queue.get()
                print(
                    f"{self.__class__.__name__}: Received response {response}. Will process if not outdated."
                )
                if response.is_draft:
                    to_draft = True
                else:
                    to_verify_semaphore += 1
                    print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
                if response.id not in self.id_to_mask:
                    print(
                        f"{self.__class__.__name__}: Response {response.id} is not in id_to_mask. Dropping."
                    )
                    self.response_queue.task_done()
                    if to_draft or (to_verify_semaphore > 0):
                        print(
                            f"{self.__class__.__name__}: Breaking out the listening loop because is a request to send. ({to_draft=}, {to_verify_semaphore=})"
                        )
                        break
                    continue
                print(
                    f"{self.__class__.__name__}: Processing response {response.id}. (It is not outdated.)"
                )
                mask: torch.Tensor = self.id_to_mask.pop(response.id)
                if response.is_draft:
                    self.draft_scores[0, mask] = response.scores
                    # scores_padded = torch.full_like(self.draft_scores[0, mask], -1)
                    n = response.scores.shape[1]
                    # scores_padded[:n] = response.scores
                    # self.draft_scores[0, mask] = scores_padded
                    self.draft_tok_ids[0, mask] = response.tok_ids[
                        0, -n:
                    ]
                    print(
                        f"{self.__class__.__name__}: Updated draft tok_ids and scores with response {response.id}. After the update, the draft tok_ids are\n{self.draft_tok_ids}"
                    )
                    mask_verified = self.tok_ids[0, mask] != -1
                    if (
                        self.tok_ids[0, mask][mask_verified]
                        != self.draft_tok_ids[0, mask][mask_verified]
                    ).any():
                        print(
                            f"{self.__class__.__name__}: The draft response {response.id} covers positions that were already verified. The draft token ids differ from the verified ones. (Draft tok_ids: {self.draft_tok_ids[0, mask]}, verified tok_ids: {self.tok_ids[0, mask]})"
                        )
                        any_rejected = True
                        self.response_queue.task_done()
                        break
                else:
                    tok_ids, any_rejected = self.rejection_sampler(response, mask)
                    tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
                    tok_ids_padded[: len(tok_ids)] = tok_ids
                    self.tok_ids[0, mask] = tok_ids_padded
                self.response_queue.task_done()
                break
            if any_rejected:
                print(
                    f"{self.__class__.__name__}: Rejected response {response.id}. Preempting all workers and resetting."
                )
                await self.preempt_all()
                self._reset()
                to_draft = True

    @torch.no_grad()
    async def send_reqeust_verify(self) -> None:
        # Select n based on the number of draft tokens waiting for verification
        mask_draft_tok_ids_to_verify = (self.tok_ids == -1) & (self.draft_tok_ids != -1)
        print(
            f"{self.__class__.__name__}: number of draft tokens waiting for verification: {mask_draft_tok_ids_to_verify.sum()}"
        )
        n = 1 + max(0, mask_draft_tok_ids_to_verify.sum())
        await self._send(
            Request.create(self.get_tok_ids_with_drafts(), n=n),
            self.verify_queue,
        )

    @torch.no_grad()
    async def send_request_draft(self) -> None:
        mask_draft_tok_ids_to_draft = (self.tok_ids == -1) & (self.draft_tok_ids == -1)
        curr_lookahead: int = min(self.lookahead, mask_draft_tok_ids_to_draft.sum() - 1)
        if curr_lookahead > 0:
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                self.draft_queue,
            )

    @torch.no_grad()
    def rejection_sampler(
        self, response: Response, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        print(
            f"{self.__class__.__name__}: Running an exact match check for response {response.id}."
        )
        response_len = response.tok_ids.shape[1]
        print(f"{self.__class__.__name__}: The response has length {response_len}.")
        tok_ids_accepted = response.tok_ids.clone()[0, mask[:response_len]]
        draft_tok_ids = self.draft_tok_ids[0, mask]
        mask_drafts_available = draft_tok_ids != -1
        any_rejected = (
            draft_tok_ids[mask_drafts_available]
            != tok_ids_accepted[mask_drafts_available]
        ).any()
        print(
            f"{self.__class__.__name__}: Comparing draft tok_ids\n{draft_tok_ids}\nwith accepted tok_ids\n{tok_ids_accepted}\nresult:\n{draft_tok_ids == tok_ids_accepted}"
        )
        if any_rejected:
            idx_first_rejected = (draft_tok_ids != tok_ids_accepted).nonzero()[0].item()
            print(
                f"{self.__class__.__name__}: First rejected token is at index {idx_first_rejected}. Accepting the first {idx_first_rejected} tokens."
            )
            tok_ids_accepted = tok_ids_accepted[: idx_first_rejected + 1]
        print(
            f"{self.__class__.__name__}: Accepting new tokens. The number of accepted tokens is {len(tok_ids_accepted)}, and the tok_ids are\n{tok_ids_accepted}"
        )
        return tok_ids_accepted, any_rejected

    def get_tok_ids_with_drafts(self) -> torch.Tensor:
        ret: torch.Tensor = self.draft_tok_ids.clone()
        nonempty_mask = self.tok_ids != -1
        ret[nonempty_mask] = self.tok_ids[nonempty_mask]
        return ret


class ManagerSI(Manager):
    async def run(self) -> None:
        print(f"{self.__class__.__name__}: Starting run")
        print(
            f"{self.__class__.__name__}: prompt's tok_ids.shape: {self.tok_ids.shape}"
        )
        print(f"{self.__class__.__name__}: prompt's tok_ids:\n{self.tok_ids}")
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(
                f"{self.__class__.__name__}: number of empty tok_ids: {(self.tok_ids == -1).sum()}"
            )
            print(f"{self.__class__.__name__}: {self.tok_ids=}")
            # 1. Draft
            mask_draft_tok_ids_to_draft = (self.tok_ids == -1) & (
                self.draft_tok_ids == -1
            )
            curr_lookahead: int = min(
                self.lookahead, mask_draft_tok_ids_to_draft.sum() - 1
            )
            if curr_lookahead > 0:
                await self._send(
                    Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                    self.draft_queue,
                )
                print(f"{self.__class__.__name__}: Waiting for draft response")
                response_draft: Response = await self.response_queue.get()
                print(
                    f"{self.__class__.__name__}: Received draft response {response_draft}."
                )
                mask: torch.Tensor = self.id_to_mask.pop(response_draft.id)
                self.draft_scores[0, mask] = response_draft.scores
                # scores_padded = torch.full_like(self.draft_scores[0, mask], -1)
                n = response_draft.scores.shape[1]
                # scores_padded[:n] = response_draft.scores
                # self.draft_scores[0, mask] = scores_padded
                self.draft_tok_ids[0, mask] = response_draft.tok_ids[0, -n:]
                print(
                    f"{self.__class__.__name__}: Updated draft tok_ids and scores with response {response_draft.id}. After the update, the draft tok_ids are\n{self.draft_tok_ids}"
                )
                self.response_queue.task_done()
            # 2. Verify
            mask_draft_tok_ids_to_verify = (self.tok_ids == -1) & (
                self.draft_tok_ids != -1
            )
            print(
                f"{self.__class__.__name__}: number of draft tokens waiting for verification: {mask_draft_tok_ids_to_verify.sum()}"
            )
            n = 1 + max(0, mask_draft_tok_ids_to_verify.sum())
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), n=n),
                self.verify_queue,
            )
            response_verify: Response = await self.response_queue.get()
            print(
                f"{self.__class__.__name__}: Received verify response {response_verify}."
            )
            mask: torch.Tensor = self.id_to_mask.pop(response_verify.id)
            tok_ids: torch.Tensor
            any_rejected: bool
            tok_ids, any_rejected = self.rejection_sampler(response_verify, mask)
            tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
            tok_ids_padded[: len(tok_ids)] = tok_ids
            self.tok_ids[0, mask] = tok_ids_padded
            self.response_queue.task_done()
            if any_rejected:
                print(
                    f"{self.__class__.__name__}: Rejected verify response {response_verify.id}."
                )
                self._reset()


class ManagerNonSI(Manager):
    async def run(self) -> None:
        print(f"{self.__class__.__name__}: Starting run")
        print(
            f"{self.__class__.__name__}: prompt's tok_ids.shape: {self.tok_ids.shape}"
        )
        print(f"{self.__class__.__name__}: prompt's tok_ids:\n{self.tok_ids}")
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(
                f"{self.__class__.__name__}: number of empty tok_ids: {(self.tok_ids == -1).sum()}"
            )
            print(f"{self.__class__.__name__}: {self.tok_ids=}")
            # # 1. Draft
            # mask_draft_tok_ids_to_draft = (self.tok_ids == -1) & (
            #     self.draft_tok_ids == -1
            # )
            # curr_lookahead: int = min(
            #     self.lookahead, mask_draft_tok_ids_to_draft.sum() - 1
            # )
            # if curr_lookahead > 0:
            #     await self._send(
            #         Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
            #         self.draft_queue,
            #     )
            #     print(f"{self.__class__.__name__}: Waiting for draft response")
            #     response_draft: Response = await self.response_queue.get()
            #     print(
            #         f"{self.__class__.__name__}: Received draft response {response_draft}."
            #     )
            #     mask: torch.Tensor = self.id_to_mask.pop(response_draft.id)
            #     self.draft_scores[0, mask] = response_draft.scores
            #     self.draft_tok_ids[0, mask] = response_draft.tok_ids[
            #         0, -response_draft.scores.shape[1] :
            #     ]
            #     print(
            #         f"{self.__class__.__name__}: Updated draft tok_ids and scores with response {response_draft.id}. After the update, the draft tok_ids are {self.draft_tok_ids}"
            #     )
            #     self.response_queue.task_done()
            # 2. Verify
            # mask_draft_tok_ids_to_verify = (self.tok_ids == -1) & (
            #     self.draft_tok_ids != -1
            # )
            # print(
            #     f"{self.__class__.__name__}: number of draft tokens waiting for verification: {mask_draft_tok_ids_to_verify.sum()}"
            # )
            # n = 1 + max(0, mask_draft_tok_ids_to_verify.sum())
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), n=1),
                self.verify_queue,
            )
            response_verify: Response = await self.response_queue.get()
            print(
                f"{self.__class__.__name__}: Received verify response {response_verify}."
            )
            mask: torch.Tensor = self.id_to_mask.pop(response_verify.id)
            tok_ids: torch.Tensor
            any_rejected: bool
            tok_ids, any_rejected = self.rejection_sampler(response_verify, mask)
            tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
            tok_ids_padded[: len(tok_ids)] = tok_ids
            self.tok_ids[0, mask] = tok_ids_padded
            self.response_queue.task_done()
            if any_rejected:
                print(
                    f"{self.__class__.__name__}: Rejected verify response {response_verify.id}."
                )
                self._reset()