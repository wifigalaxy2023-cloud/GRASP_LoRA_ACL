import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Optional

import math

import numpy as np

import torch
import torch.distributed as dist
from llama_recipes.model_checkpointing import save_model_checkpoint
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import profile, save_to_json
from llama_recipes.utils.train_utils_merge import (
    evaluation, print_and_save_log, save_model_and_optimizer_sharded,
    save_train_params)
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from tqdm import tqdm

sys.path += ["/workspace/merge/"]

from lora_allocator import LORAALLOCATORDICT


def _tokenize_for_bleu(text: str) -> List[str]:
    if text is None:
        return []
    return text.strip().split()


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu_score(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    if not predictions or len(predictions) != len(references):
        return 0.0

    max_n = max(1, max_n)
    matches = [0] * max_n
    possible = [0] * max_n
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize_for_bleu(pred)
        ref_tokens = _tokenize_for_bleu(ref)
        pred_len += max(len(pred_tokens), 1)
        ref_len += max(len(ref_tokens), 1)
        for n in range(1, max_n + 1):
            pred_counts = _count_ngrams(pred_tokens, n)
            ref_counts = _count_ngrams(ref_tokens, n)
            possible[n - 1] += sum(pred_counts.values())
            matches[n - 1] += sum(min(count, ref_counts.get(ng, 0)) for ng, count in pred_counts.items())

    precisions = []
    for match, total in zip(matches, possible):
        precisions.append((match + 1) / (total + 1))  # add-one smoothing

    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / max(pred_len, 1)))
    log_precision = sum(math.log(p) for p in precisions) / max_n
    return float(bp * math.exp(log_precision))


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model




class GRPOController:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        train_config,
        microval_loader,
        device,
        log_fn,
        rank: int,
    ):
        self.train_config = train_config
        self.config = getattr(train_config, "grpo_config", None)
        self.enabled = bool(self.config and getattr(self.config, "enable", False))
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.microval_loader = microval_loader
        self.has_microval = microval_loader is not None
        self.device = device
        self.log_fn = log_fn
        self.rank = rank or 0
        self.is_main = (not train_config.enable_fsdp) or self.rank == 0
        self.history_path = getattr(train_config, "grpo_history_path", os.path.join(train_config.output_dir, "grpo_history.json"))
        self.current_p = float(getattr(self.config, "p_init", 0.0))
        span = max(1e-3, (getattr(self.config, "p_max", 0.0) - getattr(self.config, "p_min", 0.0)) / 6.0)
        self.policy_mean = float(self.current_p)
        self.policy_std = float(span)
        self.policy_lr = float(getattr(self.config, "policy_lr", 0.05))
        self.kl_coeff = float(getattr(self.config, "kl_coeff", 0.05))
        self.entropy_bonus = float(getattr(self.config, "entropy_bonus", 0.01))
        self.max_delta_p = float(getattr(self.config, "max_delta_p", 0.1))
        self.p_min = float(getattr(self.config, "p_min", 0.0))
        self.p_max = float(getattr(self.config, "p_max", 1.0))
        self.group_size = int(max(1, getattr(self.config, "group_size", 1)))
        interval = int(getattr(self.config, "update_interval", 0))
        self.update_interval = max(1, interval)
        self.next_update_step = self.update_interval
        self.round_index = 0
        seed = int(getattr(self.config, "seed", self.train_config.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._current_metric: Optional[float] = None
        self._importance_cache: Optional[Dict[str, torch.Tensor]] = None
        self._trial_backup: Optional[Dict[str, torch.Tensor]] = None
        self._input_norm_scale: Optional[float] = None
        self._last_loss: Optional[float] = None
        self.param_list = [(name, param) for name, param in model.named_parameters() if "lora_" in name]
        self.committed_mask: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, dtype=torch.bool) for name, param in self.param_list
        }
        self.stash: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, dtype=param.dtype) for name, param in self.param_list
        }
        if self.enabled and self.is_main:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            with open(self.history_path, "w") as f:
                pass

    def should_run_round(self, optimizer_step: int) -> bool:
        if not self.enabled:
            return False
        return optimizer_step >= self.next_update_step

    def after_optimizer_step(self):
        if not self.enabled:
            return
        self._enforce_committed_mask()

    def current_val_metric(self, force: bool = False) -> Optional[float]:
        if not self.enabled:
            return None
        if not self.has_microval:
            return None
        if self._current_metric is None or force:
            self._current_metric = self._run_micro_evaluation()
        return self._current_metric

    def run_round(self, optimizer_step: int, global_step: int):
        if not self.enabled:
            return

        if not self.is_main:
            self._sync_from_main()
            self.next_update_step = optimizer_step + self.update_interval
            return

        if not self.has_microval:
            self._log("[GRPO] micro-validation loader not available; skipping update interval.")
            self._broadcast_commit(None)
            self.next_update_step = optimizer_step + self.update_interval
            return

        self.round_index += 1
        base_metric = self.current_val_metric(force=True) or 0.0
        base_loss = self._last_loss if self._last_loss is not None else None
        importance = self._compute_importance()
        candidates = self._sample_candidates()
        rewards: List[float] = []
        masks: List[Dict[str, torch.Tensor]] = []
        candidate_losses: List[float] = []

        for cand in candidates:
            mask = self._build_mask_from_importance(importance, cand)
            self._apply_mask(mask, commit=False)
            metric = self._run_micro_evaluation()
            reward = (metric or 0.0) - base_metric
            rewards.append(reward)
            masks.append(mask)
            candidate_losses.append(self._last_loss if self._last_loss is not None else float("nan"))
            self._revert_mask()

        committed_p = None
        if rewards:
            self._update_policy(candidates, rewards)
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            if rewards[best_idx] >= 0.0:
                target = self._bounded_step(candidates[best_idx])
                best_mask = masks[best_idx]
                self._apply_mask(best_mask, commit=True)
                committed_p = target
                self.current_p = target
                self._current_metric = None  # force recompute next round
            else:
                self._log("[GRPO] skip commit due to negative reward")

        self._log_round(
            optimizer_step,
            base_metric,
            base_loss,
            candidates,
            rewards,
            candidate_losses,
            committed_p,
        )
        self._broadcast_commit(committed_p)
        self.next_update_step = optimizer_step + self.update_interval

    def _enforce_committed_mask(self):
        with torch.no_grad():
            for name, param in self.param_list:
                mask = self.committed_mask[name]
                if mask.any():
                    param.data[mask] = 0

    def _compute_importance(self) -> Dict[str, torch.Tensor]:
        if self._importance_cache is not None:
            return self._importance_cache
        scale = self._estimate_input_norm()
        importance = {}
        with torch.no_grad():
            for name, param in self.param_list:
                importance[name] = param.detach().abs() * scale
        self._importance_cache = importance
        return importance

    def _estimate_input_norm(self) -> float:
        if self._input_norm_scale is not None:
            return self._input_norm_scale
        if not self.has_microval or self.microval_loader is None:
            self._input_norm_scale = 1.0
            return self._input_norm_scale
        base_model = _unwrap_model(self.model)
        norms = []
        with torch.no_grad():
            for batch in self.microval_loader:
                input_ids = batch["input_ids"].to(self.device)
                embeddings = base_model.model.embed_tokens(input_ids)
                norms.append(embeddings.norm(dim=-1).mean())
        if not norms:
            self._input_norm_scale = 1.0
        else:
            stacked = torch.stack([n.to(self.device) for n in norms])
            self._input_norm_scale = float(stacked.mean().item())
        return self._input_norm_scale

    def _build_mask_from_importance(self, importance: Dict[str, torch.Tensor], ratio: float) -> Dict[str, torch.Tensor]:
        mask_dict = {}
        ratio = max(self.p_min, min(self.p_max, ratio))
        with torch.no_grad():
            for name, tensor in importance.items():
                flat = tensor.view(-1)
                total = flat.numel()
                k = int(math.floor(ratio * total))
                if k <= 0:
                    mask = torch.zeros_like(tensor, dtype=torch.bool)
                elif k >= total:
                    mask = torch.ones_like(tensor, dtype=torch.bool)
                else:
                    threshold = torch.kthvalue(flat, k).values
                    mask = tensor <= threshold
                mask_dict[name] = mask
        return mask_dict

    def _apply_mask(self, mask_dict: Dict[str, torch.Tensor], commit: bool):
        with torch.no_grad():
            backup = {}
            for name, param in self.param_list:
                backup[name] = param.data.clone()
                mask = mask_dict[name].to(param.device)
                if commit:
                    old_mask = self.committed_mask[name]
                    newly_masked = mask & ~old_mask
                    unmasked = old_mask & ~mask
                    if newly_masked.any():
                        self.stash[name][newly_masked] = param.data[newly_masked].clone()
                    if unmasked.any():
                        param.data[unmasked] = self.stash[name][unmasked]
                        self.stash[name][unmasked] = 0
                    param.data[mask] = 0
                    self.committed_mask[name] = mask.clone()
                    if newly_masked.any():
                        self._zero_optimizer_state(param, newly_masked)
                else:
                    revive = (~mask) & self.committed_mask[name]
                    if revive.any():
                        param.data[revive] = self.stash[name][revive]
                    param.data[mask] = 0
            if not commit:
                self._trial_backup = backup
            else:
                self._trial_backup = None

    def _revert_mask(self):
        if not self._trial_backup:
            return
        with torch.no_grad():
            for name, param in self.param_list:
                param.data.copy_(self._trial_backup[name])
        self._trial_backup = None

    def _zero_optimizer_state(self, param, mask_tensor):
        state = self.optimizer.state.get(param, None)
        if not state:
            return
        mask_tensor = mask_tensor.to(param.device)
        for key, value in state.items():
            if torch.is_tensor(value):
                if value.shape == mask_tensor.shape:
                    value[mask_tensor] = 0
                elif value.dim() == 0:
                    if mask_tensor.any():
                        value.zero_()
                else:
                    try:
                        flat_val = value.view(-1)
                        flat_mask = mask_tensor.view(-1)
                        flat_val[flat_mask] = 0
                    except RuntimeError:
                        pass

    def _bounded_step(self, target: float) -> float:
        delta = target - self.current_p
        delta = max(-self.max_delta_p, min(self.max_delta_p, delta))
        new_p = self.current_p + delta
        return max(self.p_min, min(self.p_max, new_p))

    def _sample_candidates(self) -> List[float]:
        candidates = []
        for _ in range(self.group_size):
            sample = random.gauss(self.policy_mean, max(self.policy_std, 1e-3))
            sample = max(self.p_min, min(self.p_max, sample))
            candidates.append(sample)
        return candidates

    def _update_policy(self, samples: List[float], rewards: List[float]):
        if not samples or not rewards:
            return
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]
        denom = max(self.policy_std ** 2, 1e-4)
        grad_mu = sum(a * (s - self.policy_mean) / denom for a, s in zip(advantages, samples)) / len(samples)
        grad_sigma = sum(a * (((s - self.policy_mean) ** 2) / denom - 1) for a, s in zip(advantages, samples)) / len(samples)

        self.policy_mean += self.policy_lr * (grad_mu - self.kl_coeff * (self.policy_mean - self.current_p))
        self.policy_std = max(
            1e-3,
            self.policy_std + self.policy_lr * grad_sigma + self.entropy_bonus,
        )
        self.policy_mean = max(self.p_min, min(self.p_max, self.policy_mean))

    def _run_micro_evaluation(self) -> Optional[float]:
        if not self.has_microval or self.microval_loader is None:
            return None
        model_was_train = self.model.training
        base_model = _unwrap_model(self.model)
        base_model.eval()
        total_loss = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for batch in self.microval_loader:
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                input_ids = batch["input_ids"]
                outputs = base_model(**batch, use_cache=False)
                loss = outputs.loss
                weight = input_ids.size(0)
                total_loss += float(loss.item()) * weight
                total_weight += weight
        if model_was_train:
            base_model.train()
        if total_weight == 0:
            self._last_loss = None
            return 0.0
        avg_loss = total_loss / total_weight
        self._last_loss = avg_loss
        return -avg_loss

    def _log_round(self, optimizer_step, base_metric, base_loss, candidates, rewards, candidate_losses, committed_p):
        kept_fraction = 1.0 - self.current_p
        base_loss_display = base_loss if base_loss is not None else float("nan")
        base_loss_str = "nan" if math.isnan(base_loss_display) else f"{base_loss_display:.4f}"
        trial_loss_str = ["{:.4f}".format(loss) for loss in candidate_losses]
        log_message = (
            f"[GRPO] step={optimizer_step} round={self.round_index} p={self.current_p:.3f} "
            f"base_reward={base_metric:.4f} base_loss={base_loss_str} "
            f"trials={['{:.3f}'.format(c) for c in candidates]} "
            f"trial_loss={trial_loss_str} "
            f"rewards={['{:.4f}'.format(r) for r in rewards]} "
            f"commit={committed_p if committed_p is not None else 'none'} kept={kept_fraction:.3f}"
        )
        self._log(log_message)

        if self.is_main:
            entry = {
                "step": optimizer_step,
                "p_current": round(self.current_p, 6),
                "p_trials": [round(c, 6) for c in candidates],
                "rewards": [round(r, 6) for r in rewards],
                "committed_p": None if committed_p is None else round(committed_p, 6),
                "val_reward": round(base_metric, 6),
                "val_loss": None if math.isnan(base_loss_display) else round(base_loss_display, 6),
                "trial_losses": [round(loss, 6) for loss in candidate_losses],
                "kept_fraction": round(kept_fraction, 6),
            }
            with open(self.history_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _broadcast_commit(self, committed_p: Optional[float]):
        if not dist.is_available() or not dist.is_initialized():
            return
        device = self.device
        flag = torch.tensor([1 if committed_p is not None else 0], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        ratio_tensor = torch.tensor([self.current_p], device=device)
        dist.broadcast(ratio_tensor, src=0)
        if committed_p is None:
            return
        for name, param in self.param_list:
            mask = self.committed_mask[name].to(device=device, dtype=torch.uint8).view(-1)
            dist.broadcast(mask, src=0)

    def _sync_from_main(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        device = self.device
        flag = torch.zeros(1, device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        ratio_tensor = torch.zeros(1, device=device)
        dist.broadcast(ratio_tensor, src=0)
        self.current_p = float(ratio_tensor.item())
        if flag.item() == 0:
            self._current_metric = None
            return
        mask_dict = {}
        for name, param in self.param_list:
            flat = torch.empty(param.numel(), dtype=torch.uint8, device=device)
            dist.broadcast(flat, src=0)
            mask = flat.view(param.shape).bool()
            mask_dict[name] = mask
        self._apply_mask(mask_dict, commit=True)
        self._current_metric = None

    def _log(self, message: str):
        if self.is_main:
            self.log_fn(message)


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, train_dataset, microval_dataloader=None, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []

    checkpoint_times = []
    results = {}
    training_start_time = datetime.utcnow()
    overall_start_time = time.perf_counter()
    epochs_completed = 0
    best_epoch = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = getattr(train_config, "early_stopping_patience", -1)
    min_delta = float(getattr(train_config, "early_stopping_min_delta", 0.0))
    early_stop_triggered = False
    stop_reason = ""
    stop_training = False
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = train_config.num_epochs * num_update_steps_per_epoch
    optimizer_step = 0

    is_main_rank = (not train_config.enable_fsdp) or (rank == 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if train_config.enable_fsdp and local_rank is not None else "cuda:0")
    else:
        device = torch.device("cpu")
    micro_loader_for_rank = microval_dataloader if is_main_rank else None
    grpo_controller = GRPOController(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_config=train_config,
        microval_loader=micro_loader_for_rank,
        device=device,
        log_fn=lambda msg: print_and_save_log(msg, train_config),
        rank=rank or 0,
    )
    use_grpo = grpo_controller.enabled

    loraallocator = None
    if not use_grpo:
        loraallocator = LORAALLOCATORDICT[train_config.allocator](
            model=model,
            ipt_threshold=train_config.ipt_threshold,
            delete_percent=train_config.delete_percent,
        )

    for epoch in range(train_config.num_epochs):
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()

                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        grpo_controller.after_optimizer_step()
                        if loraallocator:
                            loraallocator.update_allocation(model, train_dataset)
                        optimizer.zero_grad()
                        pbar.update(1)
                        optimizer_step += 1
                        if use_grpo and grpo_controller.should_run_round(optimizer_step):
                            grpo_controller.run_round(optimizer_step, total_train_steps)

                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()
            memtrace.save_stats(train_config)

        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            improved = (eval_epoch_loss + min_delta) < best_val_loss
            if train_config.save_model and improved:
                if train_config.enable_fsdp:
                    dist.barrier()

                if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                    print("=====================================================")
                    save_model_and_optimizer_sharded(model, rank, train_config)

                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if improved:
                best_val_loss = eval_epoch_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                if train_config.enable_fsdp:
                    if rank==0:
                        print_and_save_log(f"best eval loss on epoch {epoch+1} is {best_val_loss}", train_config)
                else:
                    print_and_save_log(f"best eval loss on epoch {epoch+1} is {best_val_loss}", train_config)
            else:
                if patience >= 0:
                    epochs_without_improvement += 1
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
            if patience >= 0 and epochs_without_improvement >= patience:
                early_stop_triggered = True
                stop_reason = f"no_improvement_for_{patience}_epochs"
                if not train_config.enable_fsdp or rank == 0:
                    print_and_save_log(
                        f"Early stopping triggered at epoch {epoch+1} after {epochs_without_improvement} epochs without validation loss improvement.",
                        train_config,
                    )
                stop_training = True
        else:
            epochs_without_improvement = 0
        if train_config.enable_fsdp:
            if rank==0:
                print_and_save_log(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s", train_config)
        else:
            print_and_save_log(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s", train_config)

        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
        epochs_completed = epoch + 1
        if stop_training:
            break

    if use_grpo and is_main_rank:
        final_reward = grpo_controller.current_val_metric(force=True) or 0.0
        kept_fraction = 1.0 - grpo_controller.current_p
        final_loss = grpo_controller._last_loss if grpo_controller._last_loss is not None else float("nan")
        final_loss_display = "nan" if math.isnan(final_loss) else f"{final_loss:.4f}"
        print_and_save_log(
            f"[GRPO] Final delete ratio p={grpo_controller.current_p:.3f}, final_reward={final_reward:.4f}, final_loss={final_loss_display}, kept_fraction={kept_fraction:.3f}",
            train_config,
        )

    training_end_time = datetime.utcnow()
    training_duration_seconds = time.perf_counter() - overall_start_time
    if not stop_reason:
        if max_steps_reached:
            stop_reason = "max_train_steps_reached"
        else:
            stop_reason = "max_epochs_reached"
    best_val_loss_value = None if math.isinf(best_val_loss) else float(best_val_loss)
    results["training_start_time"] = training_start_time.isoformat()
    results["training_end_time"] = training_end_time.isoformat()
    results["training_duration_seconds"] = training_duration_seconds
    results["epochs_completed"] = epochs_completed
    results["best_epoch"] = best_epoch
    results["best_val_loss"] = best_val_loss_value
    results["early_stopping_triggered"] = early_stop_triggered
    results["stop_reason"] = stop_reason
    results["delete_percent"] = getattr(train_config, "delete_percent", None)
    results["num_epochs_requested"] = train_config.num_epochs
    results["early_stopping_patience"] = patience

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    if (not train_config.enable_fsdp) or (rank == 0):
        metadata_path = os.path.join(train_config.output_dir, "run_metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(results, metadata_file, indent=2)
        except OSError as exc:
            print_and_save_log(f"Failed to write run metadata to {metadata_path}: {exc}", train_config)
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results
