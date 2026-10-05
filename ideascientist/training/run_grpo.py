#!/usr/bin/env python
"""Generic GRPO training driver for the three sub-agent roles.

Registers ONE agent's single-phase multi-turn environment and runs nemorl's GRPO
loop. Shared by gap_finder / innovator / report_writer — each agent's thin
``run_grpo_<agent>.py`` sets up sys.path for its overlay and calls ``main(...)``
with its names, so a fix here applies to all three.

The two rollout patches (split/step context + judge-parse-failure drop) live here
verbatim so ``code/train/nemorl/`` stays unmodified.
"""
import argparse
import importlib
import os
import pprint
import sys

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.utils import ENV_REGISTRY
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def _select_trainer(master_config: MasterConfig):
    dp_cfg = master_config.data_plane or {}
    if dp_cfg.get("enabled", False):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync
        print("Running synchronous GRPO training (TransferQueue)")
        return grpo_train_sync
    print("Running synchronous GRPO training (legacy)")
    return grpo_train


def _install_rollout_context_patch(task_to_env, val_task_to_env, judge_sentinel):
    """Tell each env actor which (split, step) it is rolling out, and drop
    judge-parse failures from the batch.

    nemorl never passes the step number to the rollout entrypoint; only the
    driver loop holds it. Split comes from dict identity — the dict passed *is*
    the val dict — and the step from a module-global counter that validation
    reads without advancing.

    Both the sync and async-engine entrypoints are wrapped. One rollout call
    still shares one (split, step): grpo_train blocks on the call, so only the
    samples within it overlap. The Layer-2 asynchronous trainer is a different
    problem — see :func:`_install_async_grpo_patches`.
    """
    import nemo_rl.experience.rollouts as _rollouts_mod
    import nemo_rl.algorithms.grpo as _grpo_mod
    import ray

    # train_step: incremented per train rollout call (Layer-1 sync path).
    # val_step: the REAL validation step, captured by the validate() wrapper below.
    # Under Layer-2 async, train rollouts run in the collector and never reach this
    # wrapper, so train_step stays 0 there (train dirs are labeled by the collector-
    # side target_weight_version stamp instead); only the val path uses this wrapper.
    _state = {"train_step": 0, "val_step": 0}
    _val_env_dict_id = id(val_task_to_env) if val_task_to_env is not None else None

    def _stamp_extra_env_info(args, kwargs, step) -> None:
        """Copy-on-write stamp target_weight_version=step onto the rollout batch's
        per-sample extra_env_info, so _persist_rollout keys the rows by the real
        step (uniform with the async train path's collector-side stamp)."""
        try:
            ib = kwargs.get("input_batch")
            if ib is None and len(args) >= 2:
                ib = args[1]
            if ib is None:
                return
            eei = ib["extra_env_info"]
            for i in range(len(eei)):
                if isinstance(eei[i], dict):
                    d = dict(eei[i])
                    d["target_weight_version"] = int(step)
                    eei[i] = d
        except Exception as exc:  # noqa: BLE001 — staleness accounting degrades, training continues
            print(f"[weight-version] step {step}: {exc!r}")

    def _set_ctx(env_dict, split, step):
        if not env_dict:
            return
        seen = set()
        refs = []
        for env in env_dict.values():
            if id(env) in seen:
                continue
            seen.add(id(env))
            setter = getattr(env, "set_rollout_context", None)
            if setter is None:
                continue
            try:
                refs.append(setter.remote(split, step))
            except Exception as exc:  # noqa: BLE001 — a dead actor must not kill the run
                print(f"[rollout-context] {split} step {step}: {exc!r}")
        if refs:
            try:
                ray.get(refs)
            except Exception as exc:  # noqa: BLE001 — see above
                print(f"[rollout-context] step {step}: {exc!r}")

    def _make_wrapped(_orig):
        def _wrapped(*args, **kwargs):
            if "task_to_env" in kwargs:
                env_dict = kwargs["task_to_env"]
            elif len(args) >= 4:
                env_dict = args[3]
            else:
                env_dict = None
            is_val = (
                _val_env_dict_id is not None
                and env_dict is not None
                and id(env_dict) == _val_env_dict_id
            )
            if is_val:
                split, step = "val", _state["val_step"]
                # Stamp the real val step onto extra_env_info (belt-and-suspenders
                # alongside the _set_ctx push; val is serial so both are correct).
                _stamp_extra_env_info(args, kwargs, step)
            else:
                _state["train_step"] += 1
                split, step = "train", _state["train_step"]
            _set_ctx(env_dict, split, step)
            batch, rollout_metrics = _orig(*args, **kwargs)
            _drop_judge_failed_rollouts(batch, rollout_metrics, judge_sentinel)[0]
            return batch, rollout_metrics

        return _wrapped

    # Wrap the sync entrypoint (Layer-1 disabled) and the async entrypoint
    # (Layer-1 enabled). grpo.py imported both as module-level names
    # (grpo.py), so we must rebind them on the grpo module too — that is
    # the reference grpo_train actually calls.
    _sync_wrapped = _make_wrapped(_rollouts_mod.run_multi_turn_rollout)
    _async_wrapped = _make_wrapped(_rollouts_mod.run_async_multi_turn_rollout)
    _rollouts_mod.run_multi_turn_rollout = _sync_wrapped
    _rollouts_mod.run_async_multi_turn_rollout = _async_wrapped
    if hasattr(_grpo_mod, "run_multi_turn_rollout"):
        _grpo_mod.run_multi_turn_rollout = _sync_wrapped
    if hasattr(_grpo_mod, "run_async_multi_turn_rollout"):
        _grpo_mod.run_async_multi_turn_rollout = _async_wrapped

    # Wrap validate() to capture the REAL validation step. validate() is a driver-
    # side module function called by BOTH grpo_train (sync) and async_grpo_train
    # with step=<real step>; rebinding the name on the grpo module reaches both
    # loops (they resolve it at call time). Without this, val rollouts inherited the
    # stale _state["train_step"] (never incremented under async) and ALL landed in
    # rollouts/val/step_000/. The wrapper records the step and pushes it to the val
    # env actors (val is serial, so a mutable _cur_step is correct here).
    _orig_validate = getattr(_grpo_mod, "validate", None)
    if _orig_validate is not None and not getattr(
        _orig_validate, "_val_step_wrapped", False
    ):
        def _validate_wrapped(*a, **k):
            # signature: validate(policy_generation, val_dataloader, tokenizer,
            #   val_task_to_env, step=0, ...)
            step = k.get("step")
            if step is None and len(a) >= 5:
                step = a[4]
            _state["val_step"] = int(step) if step is not None else 0
            _set_ctx(val_task_to_env, "val", _state["val_step"])
            return _orig_validate(*a, **k)

        _validate_wrapped._val_step_wrapped = True
        _grpo_mod.validate = _validate_wrapped

    print(
        "Installed rollout-context patch (split/step + judge-drop) on BOTH "
        "sync and async rollout entrypoints"
    )


def _drop_judge_failed_rollouts(
    batch, rollout_metrics, judge_sentinel, in_place: bool = False
) -> int:
    """Zero ``loss_multiplier`` for rollouts the env flagged as judge failures.

    The reward tensor is the only per-row channel that reaches the training
    batch: EnvironmentReturn has no loss field, terminal rows' extra_env_info is
    discarded, and the env's post-process hook is never called. So the env
    encodes the failure as an out-of-range sentinel reward and it is detected
    here, before grpo turns rewards into advantages. The row's loss_multiplier
    goes to zero and its reward to 0.0 — the latter so the sentinel cannot
    distort the per-prompt baseline the other rollouts in the group are scored
    against.

    Returns ``(n_failed, failed_mask)`` so async callers can zero the decoupled
    rewards and mask they pass to compute_advantage as well.

    With ``in_place=True`` the rows are index-assigned rather than rebound:
    under async, ``train_data["sample_mask"]`` is the same tensor object as
    ``repeated_batch["loss_multiplier"]``, so replacing the tensor would leave
    that alias holding the undropped values.
    """
    import torch

    reward = batch.get("total_reward")
    if reward is None:
        return 0, None
    # A row is a judge-failure if its reward sits at/below the sentinel (the
    # sentinel is far outside the legitimate range, which is >= 0). Use a margin so
    # float round-trips through torch cannot miss it.
    failed = reward <= (judge_sentinel / 2.0)
    n_failed = int(failed.sum().item())
    if n_failed == 0:
        return 0, None

    lm = batch.get("loss_multiplier")
    if lm is not None:
        if in_place:
            lm[failed] = 0.0
        else:
            lm = lm.clone()
            lm[failed] = 0.0
            batch["loss_multiplier"] = lm

    if in_place:
        reward[failed] = 0.0
    else:
        reward = reward.clone()
        reward[failed] = 0.0
        batch["total_reward"] = reward
    # Keep any per-component reward tensors consistent with the zeroed total so
    # nothing downstream re-derives the sentinel.
    for key in list(batch.keys()):
        if key.startswith("reward") and key != "total_reward":
            comp = batch[key]
            if isinstance(comp, torch.Tensor) and comp.shape == failed.shape:
                if in_place:
                    comp[failed] = 0.0
                else:
                    comp = comp.clone()
                    comp[failed] = 0.0
                    batch[key] = comp

    if rollout_metrics is not None:
        rollout_metrics["judge_failed_dropped"] = n_failed
    print(f"Dropped {n_failed} rollout(s) from GRPO batch (judge parse failure)")
    return n_failed, failed


def _install_async_grpo_patches(task_to_env, val_task_to_env, judge_sentinel,
                                start_step: int) -> None:
    """Install the rollout context and the judge-drop for the asynchronous trainer.

    The sync wrapper cannot reach async_grpo_train: its rollouts run inside an
    ``AsyncTrajectoryCollector`` actor, which lives in its own process and bound
    the entrypoint at its own import time. A per-call step counter would be
    meaningless there anyway — the collector runs one thread per prompt group,
    concurrently and decoupled from training steps.

    Two seams do run in the driver process:

    1. **Context**, via ``refit_policy_generation``. The driver bumps
       ``weight_version`` after each refit, and the collector then produces
       trajectories targeting ``weight_version + [1..max_age]``. Mirroring that
       counter and pushing ("train", step) to the env actors after each refit
       groups persisted rollouts the same way the sync path does. Validation
       still runs through the sync ``validate()``, so val rollouts are stamped
       by the sync wrapper.

    2. **Judge-drop**, via the advantage estimator's ``compute_advantage``. That
       call is the last point where the sampled batch is visible with rewards
       and mask together and before advantages exist. The drop is applied in
       place on ``repeated_batch`` — whose tensors train_data aliases, so the
       policy loss sees it too — and to the decoupled rewards and mask the
       advantage math itself uses.
    """
    import nemo_rl.algorithms.grpo as _grpo_mod
    import ray

    # ---- (1) step/split context via refit_policy_generation ----------------
    train_envs = list((task_to_env or {}).values())

    def _push_ctx(step: int) -> None:
        refs = []
        seen = set()
        for env in train_envs:
            if id(env) in seen:
                continue
            seen.add(id(env))
            setter = getattr(env, "set_rollout_context", None)
            if setter is None:
                continue
            try:
                refs.append(setter.remote("train", step))
            except Exception as exc:  # noqa: BLE001 — a dead actor must not kill the run
                print(f"[rollout-context] train step {step}: {exc!r}")
        if refs:
            try:
                ray.get(refs)
            except Exception as exc:  # noqa: BLE001 — see above
                print(f"[rollout-context] step {step}: {exc!r}")

    # Mirror of async_grpo_train's weight_version: starts at start_step, the
    # collector's first target is start_step+1.
    _ctr = {"target_step": start_step + 1}
    _push_ctx(_ctr["target_step"])  # initial window (before first refit)

    _orig_refit = _grpo_mod.refit_policy_generation

    def _refit_wrapped(*args, **kwargs):
        out = _orig_refit(*args, **kwargs)
        # After this refit the driver bumps weight_version and the collector begins
        # generating for the NEXT target step. Advance our mirror + restamp envs.
        _ctr["target_step"] += 1
        _push_ctx(_ctr["target_step"])
        return out

    _grpo_mod.refit_policy_generation = _refit_wrapped

    # ---- (2) judge-drop via compute_advantage ------------------------------
    _orig_create = _grpo_mod._create_advantage_estimator

    def _create_wrapped(master_config):
        estimator = _orig_create(master_config)
        _orig_compute = estimator.compute_advantage

        def _compute_wrapped(prompt_ids, rewards, mask, **kwargs):
            repeated_batch = kwargs.get("repeated_batch")
            if repeated_batch is not None:
                n_failed, failed = _drop_judge_failed_rollouts(
                    repeated_batch, None, judge_sentinel, in_place=True
                )
                if failed is not None:
                    # rewards / mask are DECOUPLED copies passed positionally by the
                    # loop; zero the same rows so advantages ignore judge-failures.
                    if rewards is not None and rewards.shape == failed.shape:
                        rewards = rewards.clone()
                        rewards[failed] = 0.0
                    if mask is not None and mask.shape[0] == failed.shape[0]:
                        mask = mask.clone()
                        mask[failed] = 0.0
            return _orig_compute(prompt_ids, rewards, mask, **kwargs)

        estimator.compute_advantage = _compute_wrapped
        return estimator

    _grpo_mod._create_advantage_estimator = _create_wrapped

    print(
        "Installed async-GRPO patches: (split/step) context via refit hook "
        f"(start target step {start_step + 1}), judge-drop via compute_advantage"
    )


def _install_reward_fields_logging(logger, run_dir: str, agent_name: str) -> None:
    """Log per-field reward means to the trainer's OWN TensorBoard, in-process.

    Wraps ``logger.log_metrics`` so that on each ``prefix="train"`` / ``"validation"``
    step the driver reads that step's just-persisted rollouts
    (``<run>/rollouts/<split>/step_<NNN>/``, correctly labeled by target_weight_version)
    and merges ``reward_fields/*`` — the base/reference/citation/negative-penalty
    aggregates, every rubric field's judge score, and gate pass-rates — into the metrics
    dict. They then flow through the same TB/W&B logger as ``total``/``rubric_reward``
    under ``train/reward_fields/*`` and ``validation/reward_fields/*`` — no separate
    process, same TB run. Reuses the aggregation from ``reward_fields_tb`` (sibling
    module on PYTHONPATH). Env-gate ``IDEASCIENTIST_REWARD_FIELDS_TB=0`` to disable.
    Never breaks training: all work is wrapped in try/except.
    """
    if os.environ.get("IDEASCIENTIST_REWARD_FIELDS_TB", "1") == "0":
        return
    try:
        import reward_fields_tb as _rf
        ctx = _rf._load_agent(agent_name)
    except Exception as exc:  # setup failed -> leave logging untouched
        print(f"reward-fields logging disabled (setup failed: {exc})")
        return

    _orig = logger.log_metrics
    _seen: set = set()
    _split_by_prefix = {"train": "train", "validation": "val"}

    def _wrapped(metrics, step, prefix="", *args, **kwargs):
        split = _split_by_prefix.get(prefix)
        if split is not None and (split, step) not in _seen:
            try:
                step_dir = os.path.join(
                    run_dir, "rollouts", split, f"step_{int(step):03d}"
                )
                if os.path.isdir(step_dir):
                    agg = _rf._collect_step(step_dir, agent_name, ctx)
                    if agg:
                        _seen.add((split, step))
                        # NeMo-RL's raw reward tensor contains judge-failed rows
                        # reset to zero for numerical safety. Its unmasked mean
                        # therefore falls when infrastructure failures increase,
                        # even though those rows have loss_multiplier=0 and never
                        # train the policy. Use the persisted valid-rollout mean so
                        # train/reward matches the actual GRPO batch convention.
                        canonical_reward = agg.get("total")
                        metrics = {
                            **metrics,
                            **(
                                {"reward": canonical_reward}
                                if canonical_reward is not None
                                else {}
                            ),
                            **{f"reward_fields/{k}": v for k, v in agg.items()},
                        }
            except Exception as exc:  # never break the real metric logging
                print(f"reward-fields enrich failed ({prefix} step {step}): {exc}")
        return _orig(metrics, step, prefix=prefix, *args, **kwargs)

    logger.log_metrics = _wrapped
    print(f"Installed in-driver reward-fields logging (agent={agent_name})")




def _seed_resume_checkpoint(run_dir: str, ckpt_dir: str) -> None:
    """Per-launch checkpoint dirs: seed THIS launch's (empty) checkpoint dir with a
    COPY of the latest checkpoint from any PRIOR launch, so nemorl's
    ``get_latest_checkpoint_path()`` (which globs ``checkpoint_dir/step_*``) resumes
    from it while NEW checkpoints for this launch save here — isolating each resume's
    checkpoints instead of sharing/pruning one dir.

    Discovers the highest ``step_N`` across sibling launches
    (``run_dir/run_*/checkpoints``) and the legacy top-level (``run_dir/checkpoints``,
    for pre-change runs). No-op if this launch already has checkpoints (mid-launch
    re-entry after a crash) or none exist anywhere (fresh run). COPIES (not symlinks)
    because ``keep_top_k`` prunes with ``shutil.rmtree``, which raises on a symlinked
    directory. Runs single-process on the driver before Ray/setup, so it is race-free.
    """
    import glob
    import re
    import shutil

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir_abs = os.path.abspath(ckpt_dir)

    def _steps_in(d: str):
        found = []
        for p in glob.glob(os.path.join(d, "step_*")):
            m = re.fullmatch(r"step_(\d+)", os.path.basename(p))
            if m and os.path.isdir(p) and os.path.exists(
                os.path.join(p, "training_info.json")
            ):
                found.append((int(m.group(1)), p))
        return found

    # Already seeded / mid-launch re-entry: keep this launch's own checkpoints.
    if _steps_in(ckpt_dir):
        print(f"[resume-seed] {ckpt_dir} already has checkpoints; not seeding.")
        return

    candidates = []
    for d in glob.glob(os.path.join(run_dir, "run_*", "checkpoints")):
        if os.path.abspath(d) != ckpt_dir_abs:
            candidates += _steps_in(d)
    legacy = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(legacy) and os.path.abspath(legacy) != ckpt_dir_abs:
        candidates += _steps_in(legacy)

    if not candidates:
        print("[resume-seed] no prior checkpoint under run_dir; starting fresh.")
        return

    step, src = max(candidates, key=lambda x: x[0])
    dst = os.path.join(ckpt_dir, os.path.basename(src))
    tmp = dst + ".seeding"
    print(f"[resume-seed] seeding step_{step} from {src} -> {dst} (copy) to resume.")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)
    print(
        f"[resume-seed] done; this launch resumes at step_{step} and saves new "
        f"checkpoints under {ckpt_dir}."
    )


ROLES = {
    "gap_finder": "ideascientist.training.environments.GapFinderEnvironment",
    "innovator": "ideascientist.training.environments.InnovatorEnvironment",
    "report_writer": "ideascientist.training.environments.ReportWriterEnvironment",
}


def main(role: str) -> None:
    """Run GRPO for one sub-agent role.

    Registers the role's environment and imports its data module, which
    self-registers the sample processor. Without that processor the built-in
    one drops the per-sample metadata, the environment cannot reach the
    reference artifact, and every rollout silently scores zero.
    """
    if role not in ROLES:
        raise SystemExit(f"unknown role {role!r}; expected one of {sorted(ROLES)}")
    agent_name = role
    env_class_fqn = ROLES[role]
    data_module = "ideascientist.training.data"
    reward_module = f"ideascientist.rewards.{role}.reward"

    register_omegaconf_resolvers()

    ENV_REGISTRY[agent_name] = {"actor_class_fqn": env_class_fqn}
    ACTOR_ENVIRONMENT_REGISTRY[env_class_fqn] = PY_EXECUTABLES.SYSTEM

    importlib.import_module(data_module)

    # The sentinel the rollout patch uses to drop rollouts the judge could not score.
    judge_sentinel = importlib.import_module(reward_module).JUDGE_FAILED_REWARD_SENTINEL

    parser = argparse.ArgumentParser(description=f"GRPO training for {agent_name}")
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    print("Final config:")
    pprint.pprint(config)

    # RUN_DIR is the stable root passed across resumes. Per-launch outputs (logs,
    # rollouts, TensorBoard, config snapshot) always go into a NEW timestamped
    # subfolder RUN_DIR/run_<ts>/. Checkpoints go either:
    #   * (DEFAULT, IDEASCIENTIST_PER_LAUNCH_CHECKPOINTS != 0) launch_dir/checkpoints
    #     — ISOLATED per launch. The new launch's dir is seeded with a COPY of the
    #     latest prior checkpoint (across sibling launches / legacy top-level) so it
    #     still resumes, while its new checkpoints + keep_top_k pruning NEVER touch a
    #     prior launch's dir. So a resume cannot delete the run it resumed from, and
    #     keep_top_k applies INDEPENDENTLY per launch (not cumulatively across the
    #     initial + resume + resume-of-resume chain). This is the default because the
    #     shared-dir mode below pruned a still-needed checkpoint (step_15) once.
    #   * (IDEASCIENTIST_PER_LAUNCH_CHECKPOINTS=0) RUN_DIR/checkpoints — SHARED across
    #     launches; simplest, but each resume mixes into and keep_top_k-prunes the one
    #     shared dir (can delete an earlier launch's checkpoints). Opt-in only.
    run_dir = os.environ.get("RUN_DIR", "").strip()
    if not run_dir:
        import datetime
        base = os.path.dirname(config.logger["log_dir"].rstrip("/")) or "."
        run_dir = os.path.join(base, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    # Per-launch subfolder. The launcher passes LAUNCH_DIR (so its log lands in
    # the SAME subfolder); ad-hoc launches synthesize one here.
    launch_dir = os.environ.get("LAUNCH_DIR", "").strip()
    if not launch_dir:
        import datetime
        launch_dir = os.path.join(
            run_dir, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(launch_dir, exist_ok=True)
    config.logger["log_dir"] = os.path.join(launch_dir, "logs")
    if os.environ.get("IDEASCIENTIST_PER_LAUNCH_CHECKPOINTS", "1").strip() != "0":
        ckpt_dir = os.path.join(launch_dir, "checkpoints")
        config.checkpointing["checkpoint_dir"] = ckpt_dir
        _seed_resume_checkpoint(run_dir, ckpt_dir)
    else:
        config.checkpointing["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    config.env.setdefault(agent_name, {})
    config.env[agent_name]["output_dir"] = launch_dir
    # Optional run tag (IDEASCIENTIST_RUN_TAG, e.g. "compare") to distinguish reward
    # variants in TensorBoard + W&B: it suffixes the central-TB grouping label
    # (innovator -> innovator-compare) and the W&B run name, so a variant's curves are
    # not conflated with the baseline's. Empty = no tag (baseline behavior).
    run_tag = os.environ.get("IDEASCIENTIST_RUN_TAG", "").strip()
    tb_label = f"{agent_name}-{run_tag}" if run_tag else agent_name
    if run_tag:
        _wb = config.logger.get("wandb")
        if isinstance(_wb, dict) and _wb.get("name"):
            _wb["name"] = f"{_wb['name']}-{run_tag}"
    print(f"Run directory (root): {run_dir}")
    print(f"Launch directory (logs/rollouts/tb this launch): {launch_dir}")
    print(f"Checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    try:
        with open(os.path.join(launch_dir, "config.snapshot.yaml"), "w") as _fh:
            OmegaConf.save(config=OmegaConf.create(config.model_dump()), f=_fh)
    except Exception as _e:  # snapshot is best-effort; never block the run
        print(f"WARN: could not write config.snapshot.yaml: {_e}")

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    # Aggregate this run's TensorBoard events into the central tree so a single
    # `tensorboard --logdir <repo>/logs/tensorboard` sees every agent's runs. Pass
    # launch_dir so each launch is a distinct run in the central tree (its basename
    # is the launch timestamp).
    if config.logger.get("tensorboard_enabled"):
        _common = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "common")
        )
        if _common not in sys.path:
            sys.path.insert(0, _common)
        from tb_central import link_tensorboard_central

        _tb = link_tensorboard_central(config.logger["log_dir"], tb_label, launch_dir)
        if _tb:
            print(f"TensorBoard (central): {_tb}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env,
    )

    _dp_cfg = config.data_plane or {}
    if _dp_cfg.get("enabled", False):
        from nemo_rl.models.policy.tq_policy import TQPolicy
        _policy_factory = lambda **kwargs: TQPolicy(**kwargs, dp_cfg=_dp_cfg)
    else:
        _policy_factory = None

    (
        policy, policy_generation, _nemo_gym, cluster,
        dataloader, val_dataloader, loss_fn, logger,
        checkpointer, grpo_state, master_config,
    ) = setup(config, tokenizer, dataset, val_dataset, policy_factory=_policy_factory)

    _install_rollout_context_patch(task_to_env, val_task_to_env, judge_sentinel)
    # rollouts are persisted under launch_dir/rollouts (output_dir), so the in-driver
    # per-field logger must read from launch_dir, not the RUN_DIR top level.
    _install_reward_fields_logging(logger, launch_dir, agent_name)

    # Layer-2 asynchronous GRPO: overlaps step N+1 generation with step N backward
    # (total step time ~= max(rollout, backward) instead of their sum). Built into
    # nemo_rl but not selected by _select_trainer — dispatch it explicitly here,
    # mirroring examples/run_grpo.py's async branch (guards + real signature).
    _async_cfg = config.grpo.get("async_grpo", {}) if "async_grpo" in config.grpo else {}
    if _async_cfg.get("enabled", False):
        # Async GRPO does not support dynamic sampling / reward scaling / reward
        # shaping (DAPO features) or multiple dataloaders — fail loudly (parity with
        # examples/run_grpo.py) rather than silently mis-train.
        if config.grpo.get("use_dynamic_sampling", False):
            raise NotImplementedError("use_dynamic_sampling is not supported with async GRPO")
        for _feature in ("reward_scaling", "reward_shaping"):
            _fcfg = config.grpo.get(_feature)
            if isinstance(_fcfg, dict) and _fcfg.get("enabled", False):
                raise NotImplementedError(f"{_feature} is not supported with async GRPO")
        if config.data.get("use_multiple_dataloader", False):
            raise NotImplementedError("use_multiple_dataloader is not supported with async GRPO")

        from nemo_rl.algorithms.grpo import async_grpo_train

        # Install the async-specific (split, step) context + judge-drop seams (the
        # driver-process hooks; the collector actor is out-of-process). start_step =
        # resume step so the mirrored target-weight counter is aligned on resume.
        _install_async_grpo_patches(
            task_to_env, val_task_to_env, judge_sentinel,
            start_step=int(grpo_state["current_step"]),
        )

        print("🚀 Running async GRPO training")
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=_async_cfg["max_trajectory_age_steps"],
        )
        return

    trainer = _select_trainer(master_config)
    trainer(
        policy, policy_generation,
        dataloader, val_dataloader, tokenizer, loss_fn,
        task_to_env, val_task_to_env,
        logger, checkpointer, grpo_state, master_config,
    )


if __name__ == "__main__":
    import sys

    role_arg = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
    if role_arg:
        sys.argv.remove(role_arg)
    main(role_arg)
