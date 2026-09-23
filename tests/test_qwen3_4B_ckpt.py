import json
import os
from argparse import ArgumentParser
from shlex import quote

import vime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")
parser.add_argument("--save-optimizer", choices=["cpu", "gpu", "nvme"], default="cpu", help="Optimizer placement for save.")
parser.add_argument("--load-optimizer", choices=["cpu", "gpu", "nvme"], default="cpu", help="Optimizer placement for load.")
parser.add_argument("--checkpoint-dir", default=None, help="Directory used for the save/load checkpoint roundtrip.")


def default_checkpoint_dir(args):
    save_mode = "async" if args.async_save else "sync"
    return f"/root/models/{MODEL_NAME}_vime_{save_mode}_{args.save_optimizer}_save_{args.load_optimizer}_load"


def prepare(checkpoint_dir: str):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"rm -rf {quote(checkpoint_dir)}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def optimizer_args(optimizer: str, checkpoint_dir: str):
    args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if optimizer == "cpu":
        args += "--use-precision-aware-optimizer "
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    elif optimizer == "gpu":
        args += "--use-precision-aware-optimizer "
    elif optimizer == "nvme":
        nvme_dir = quote(f"{checkpoint_dir}_nvme_scratch")
        args += (
            "--stream-optimizer-state-to-disk "
            f"--offload-train-disk-dir {nvme_dir} "
            "--offload-train-disk-chunk-mb 64 "
            "--stream-optimizer-state-moment-dtype fp32 "
        )
    return args


def assert_nvme_checkpoint(checkpoint_dir: str):
    manifests = sorted(Path(checkpoint_dir).rglob("manifest.json"))
    assert manifests, f"No NVMe optimizer manifests found below {checkpoint_dir}"
    bucket_files = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("dtypes", {}).get("main") == "torch.float32"
        assert manifest.get("buckets"), f"Empty NVMe bucket manifest: {manifest_path}"
        for bucket in manifest["buckets"]:
            assert (manifest_path.parent / bucket["file"]).is_file()
            bucket_files += 1
    print(f"Validated {len(manifests)} NVMe manifests and {bucket_files} bucket files")


def execute(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = ""):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    checkpoint_dir_arg = quote(checkpoint_dir)
    if mode == "save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
    elif mode == "async_save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load {checkpoint_dir_arg} "
        ckpt_args += "--ckpt-step 1 "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--variable-global-batch-size "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.8 --vllm-max-cudagraph-capture-size 16 "
    )

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args(optimizer, checkpoint_dir)} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    # TODO also use typer
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args)
    prepare(checkpoint_dir)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        "save" if not args.async_save else "async_save",
        optimizer=args.save_optimizer,
        checkpoint_dir=checkpoint_dir,
    )
    if args.save_optimizer == "nvme":
        assert_nvme_checkpoint(checkpoint_dir)
    execute("load", optimizer=args.load_optimizer, checkpoint_dir=checkpoint_dir)
