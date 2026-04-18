"""
Event-driven federated fine-tuning base classes.

Simulation clock follows real device trace timestamps.  Each client reads its
trainable windows from trace_utils at initialisation and contributes
WINDOW_OPEN / WINDOW_CLOSE events to the global queue.  Training and upload
completions are scheduled as further events, so the entire simulation is
driven by popping the earliest event from the queue.

Client lifecycle per window
───────────────────────────
  IDLE  ──(WINDOW_OPEN)──►  AVAILABLE  ──(assigned)──►  TRAINING
                                                              │
                                               (TRAINING_DONE)▼
                                                         UPLOADING
                                                              │
                                               (UPLOAD_DONE)  ▼
                                                         AVAILABLE
                                                              │
                                               (WINDOW_CLOSE) ▼
                                                           IDLE

If WINDOW_CLOSE fires while a client is TRAINING or UPLOADING the in-progress
work is discarded and the client returns to IDLE.
"""

import heapq
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, List, Optional

from peft import get_peft_model
from utils.data_utils import load_data
from utils.sys_utils import device_config
from utils.train_utils import Trainer
from utils.eval_utils import Evaluator
from alg.base import BaseClient, BaseServer
from utils.model_utils import load_model, load_tokenizer, load_lora_config
from utils.time_utils import time_record
from utils.trace_utils import get_trainable_windows

# ─────────────────────────────────────────────────────────────────────────────
# Event primitives
# ─────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    WINDOW_OPEN   = "window_open"    # client enters a trainable window (from trace)
    WINDOW_CLOSE  = "window_close"   # client exits a trainable window (from trace)
    TRAINING_DONE = "training_done"  # client finishes local training
    UPLOAD_DONE   = "upload_done"    # client finishes uploading weights to server


class ClientState(Enum):
    IDLE      = "idle"       # outside every trainable window
    AVAILABLE = "available"  # inside a window, not yet assigned work
    TRAINING  = "training"   # executing local training
    UPLOADING = "uploading"  # transferring weights to the server


@dataclass
class Event:
    """A simulation event ordered by its scheduled time."""
    time: datetime
    type: EventType
    client: Optional[Any] = field(default=None, compare=False)
    # Sequence number for stable ordering of simultaneous events
    _seq: int = field(default=0, compare=False, repr=False)

    def __lt__(self, other: "Event") -> bool:
        return (self.time, self._seq) < (other.time, other._seq)

    def __le__(self, other: "Event") -> bool:
        return (self.time, self._seq) <= (other.time, other._seq)


class EventQueue:
    """Min-heap priority queue ordered by simulated time."""

    def __init__(self) -> None:
        self._heap: List[Event] = []
        self._seq: int = 0

    def push(self, event: Event) -> None:
        event._seq = self._seq
        self._seq += 1
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class EventBaseClient(BaseClient):
    """
    Client augmented with trace-based availability.

    At construction the client loads all of its trainable windows from
    trace_utils.  The server uses these windows to populate the global
    event queue with WINDOW_OPEN / WINDOW_CLOSE events.
    """

    def __init__(self, id, args):
        super().__init__(id, args)
        self.tokenizer = load_tokenizer(args)
        self.dataset = load_data(args=args, idx=self.id)
        self.lora = {}
        self.trainer = Trainer(args=args, dataset=self.dataset, client=self)
        self.evaluator = Evaluator(args=args, dataset=self.dataset)
        self.state = ClientState.IDLE

        try:
            self.windows: List[dict] = get_trainable_windows(id)
        except (KeyError, FileNotFoundError):
            self.windows = None
            
    @time_record
    def run(self, model):
        self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        return self.evaluator.evaluate(model, round_idx=self.server.round, client_id=self.id)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_UPLOAD_BANDWIDTH_MBPS = 1.0  # fallback uplink bandwidth when not configured


class EventBaseServer(BaseServer):
    """
    Synchronous FL server driven by a global event queue.

    One call to run() processes events until sample_num clients have uploaded
    their weights and the server has aggregated them — i.e., one FL round.
    This keeps the interface compatible with main.py's outer training loop.

    Simulated time (self.wall_clock_time) advances monotonically as events are
    consumed.  Wall-clock time is reported as seconds elapsed from the
    earliest trace event across all clients.
    """

    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model = get_peft_model(load_model(args), load_lora_config(args))
        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_" in k}
        self.sample_rate = args.sr
        self.round = 0

        for client, delay in zip(clients, device_config(args)):
            client.delay = delay

        lora_bytes = sum(v.numel() * v.element_size() for v in self.global_lora.values())
        bandwidth_mbps = getattr(args, 'upload_bandwidth', _DEFAULT_UPLOAD_BANDWIDTH_MBPS)
        upload_secs = lora_bytes / (bandwidth_mbps * 1e6 / 8)
        print(f'[upload] LoRA size: {lora_bytes / 1024:.1f} KB, '
              f'bandwidth: {bandwidth_mbps} Mbps, '
              f'estimated upload delay: {upload_secs:.2f} s')
        self.upload_delay = timedelta(seconds=upload_secs)
        self.sample_num = max(1, int(self.sample_rate * len(self.clients)))

        self.wall_clock_time: Optional[datetime] = None
        self._round_complete: bool = False
        self._uploads_this_round: List[EventBaseClient] = []
        self.sampled_clients: List[EventBaseClient] = []
        self.early_break: bool = False

        # Compute epoch from clients that have trace windows; fall back to now().
        all_starts = [w['start'] for c in clients for w in c.windows]
        epoch = min(all_starts) if all_starts else datetime.now()
        session_hours = getattr(args, 'session_time', 24)
        self._deadline: datetime = epoch + timedelta(hours=session_hours)

        # Clients with no trace data get one large window covering the full session.
        for client in clients:
            if client.windows is None:
                client.windows = [{'start': epoch, 'end': self._deadline}]
                print(f'[init] Client {client.id}: no trace data — '
                      f'using full session window ({epoch} → {self._deadline}).')

        # Global event queue — pre-populated with all availability windows
        self.event_queue = EventQueue()
        for client in clients:
            for window in client.windows:
                self.event_queue.push(Event(window['start'], EventType.WINDOW_OPEN, client))
                self.event_queue.push(Event(window['end'],   EventType.WINDOW_CLOSE, client))

    # ── public interface ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Process events until one FL round completes or the session deadline is reached."""
        self._round_complete = False
        self._uploads_this_round = []
        self.sampled_clients = random.sample(self.clients, self.sample_num)
        self.round += 1
        while self.event_queue and not self._round_complete:
            event = self.event_queue.pop()
            self.wall_clock_time = event.time
            if self.wall_clock_time > self._deadline:
                print(f'[{self.wall_clock_time}] Session deadline reached after round {self.round - 1}, stopping.')
                self.early_break = True
                break
            self._dispatch(event)

    # sample() and local_run() are handled reactively; keep stubs for ABC.
    def sample(self) -> None:
        pass

    def local_run(self) -> None:
        pass

    # ── event dispatch ────────────────────────────────────────────────────────

    def _dispatch(self, event: Event) -> None:
        _handlers = {
            EventType.WINDOW_OPEN:   self._on_window_open,
            EventType.WINDOW_CLOSE:  self._on_window_close,
            EventType.TRAINING_DONE: self._on_training_done,
            EventType.UPLOAD_DONE:   self._on_upload_done,
        }
        _handlers[event.type](event)

    def _on_window_open(self, event: Event) -> None:
        client: EventBaseClient = event.client
        client.state = ClientState.AVAILABLE
        print(f'[{self.wall_clock_time}] Client {client.id}: window opened.')
        self._try_assign(client)

    def _on_window_close(self, event: Event) -> None:
        client: EventBaseClient = event.client
        print(f'[{self.wall_clock_time}] Client {client.id}: window closed '
              f'(was {client.state.value}).')
        if client.state in (ClientState.TRAINING, ClientState.UPLOADING):
            # Discard in-progress work for this round
            if client in self._uploads_this_round:
                self._uploads_this_round.remove(client)
        client.state = ClientState.IDLE

    def _on_training_done(self, event: Event) -> None:
        client: EventBaseClient = event.client
        upload_done_at = event.time + self.upload_delay
        client.state = ClientState.UPLOADING
        
        if not self._window_closes_before_upload(client, upload_done_at):
            self.event_queue.push(Event(upload_done_at, EventType.UPLOAD_DONE, client))
        

    def _on_upload_done(self, event: Event) -> None:
        client: EventBaseClient = event.client
        client.state = ClientState.AVAILABLE
        self._uploads_this_round.append(client)
        print(f'[{self.wall_clock_time}] Client {client.id}: upload complete '
              f'({len(self._uploads_this_round)}/{len(self.sampled_clients)}).')

        if set(self._uploads_this_round) >= set(self.sampled_clients):
            self.aggregate()
            self._round_complete = True

    # ── aggregation ───────────────────────────────────────────────────────────

    def aggregate(self) -> None:
        data_sum = sum(len(c.dataset['train']) for c in self.sampled_clients)
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)
        for client in self.sampled_clients:
            for k, v in client.lora.items():
                aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum
        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'[{self.wall_clock_time}] Aggregated model updated (round {self.round}).')

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)
        res_dict = {}
        for k in all_metrics[0].keys():
            res_dict[k] = sum(m[k] for m in all_metrics) / len(all_metrics)

        return res_dict

    def save_adapter(self):
        import os
        import torch
        args = self.args
        name = (
            f'{args.alg}_{args.dataset}_{args.model}_'
            f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        )
        adapter_path = os.path.join(args.suffix, name, 'adapter')
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(self.global_lora, os.path.join(adapter_path, 'lora_weights.pt'))
        print(f'Adapter saved to {adapter_path}')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _try_assign(self, client: EventBaseClient) -> None:
        """Assign training to *client* if sampled this round and not yet uploaded."""
        if client not in self.sampled_clients:
            return  # not selected for this round

        if client in self._uploads_this_round:
            return  # already submitted an update this round

        print(f'[{self.wall_clock_time}] Assigning training to client {client.id}.')
        client.state = ClientState.TRAINING
        # Run actual training; client.training_time is set by @time_record
        client.run(self.model)
        # Restore global weights so the next client trains from the same base
        self.model.load_state_dict(self.global_lora, strict=False)
        finish_at = self.wall_clock_time + timedelta(seconds=client.training_time)
        
        if not self._window_closes_before_upload(client, finish_at):
            self.event_queue.push(Event(finish_at, EventType.TRAINING_DONE, client))
        
        
    def _window_closes_before_upload(self, client: EventBaseClient, upload_done_at: datetime) -> bool:
        return any(
            e.type == EventType.WINDOW_CLOSE and e.client is client and e.time < upload_done_at
            for e in self.event_queue._heap
        )