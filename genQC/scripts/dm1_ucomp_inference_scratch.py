# genQC/scripts/infer_dm1_discrete_unitary_compilation.py
#
# Simple inference script for the DM1 discrete unitary-compilation pipeline.
#
# What it does:
#   1) loads your saved trained pipeline
#   2) builds a target unitary from a small example Qiskit circuit
#   3) creates the text prompt used during training: "Compile using: [...]"
#   4) samples random latent noise
#   5) runs diffusion inference
#   6) decodes the generated latent tensors back into circuits
#   7) ranks the generated circuits by Frobenius distance to the target unitary
#
# IMPORTANT:
# - This is mainly for testing the inference path on your smoke-test model.
# - Since the model is only dummy-trained, the generated circuits may be poor.
# - For a real use case, replace build_example_target_circuit(...) with your own target.

from pathlib import Path
from genQC.pipeline import pipeline
import numpy as np
import torch

from qiskit import QuantumCircuit

from genQC.pipeline.compilation_diffusion_pipeline import DiffusionPipeline_Compilation
from genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from genQC.platform.backends.circuits_qiskit import CircuitsQiskitBackend


def build_example_target_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Build a small target circuit using only the same discrete gate set
    that the training script used.

    Current gate pool in training script:
        ["h", "x", "z", "cx", "ccx"]

    You can replace this with any other Qiskit circuit on the same number
    of qubits.
    """
    qc = QuantumCircuit(num_qubits)

    # Example target that stays inside the trained discrete gate set
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.z(2)

    if num_qubits >= 4:
        qc.ccx(0, 1, 3)

    return qc


def unitary_to_condition_tensor(U: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Convert a complex unitary matrix U into the training/inference condition format:
        [batch, 2, N, N]
    where channel 0 = real part, channel 1 = imaginary part.
    """
    U_r = torch.from_numpy(np.real(U)).to(torch.float32)
    U_i = torch.from_numpy(np.imag(U)).to(torch.float32)
    U_t = torch.stack([U_r, U_i], dim=0)          # [2, N, N]
    U_t = U_t.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch, 2, N, N]
    return U_t.to(device)


def build_prompt_tokens(
    pipeline: DiffusionPipeline_Compilation,
    gate_pool: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the text condition in the same style as the dataset generator:
        "Compile using: ['h', 'x', 'z', 'cx', 'ccx']"

    We pass tokenized prompts directly as a 2D tensor [batch, seq].
    That works with the pipeline's prepare_c_emb() path.
    """
    prompt = f"Compile using: {[str(g) for g in gate_pool]}"
    prompts = [prompt] * batch_size
    c = pipeline.text_encoder.tokenize_and_push_to_device(prompts, to_device=True)
    return c.to(device)


def decode_latents_to_circuits(
    pipeline: DiffusionPipeline_Compilation,
    latents_pred: torch.Tensor,
    tokenizer: CircuitTokenizer,
    backend: CircuitsQiskitBackend,
):
    """
    Convert generated continuous latent tensors back into Qiskit circuits.

    Steps:
      latent -> token tensor via embedder.invert(...)
      token tensor -> CircuitInstructions via tokenizer.decode(...)
      CircuitInstructions -> QuantumCircuit via backend.genqc_to_backend(...)
    """
    # token_tensors should be [batch, num_qubits, max_gates]
    token_tensors = pipeline.embedder.invert(latents_pred).detach().cpu()

    circuits = []
    instructions_list = []

    for i in range(token_tensors.shape[0]):
        try:
            instructions = tokenizer.decode(
                token_tensors[i],
                ignore_errors=True,
                place_error_placeholders=False,
            )
            qc = backend.genqc_to_backend(
                instructions,
                place_barriers=False,
                ignore_errors=True,
                place_error_placeholders=False,
            )
        except Exception:
            instructions = None
            qc = None

        instructions_list.append(instructions)
        circuits.append(qc)

    return token_tensors, instructions_list, circuits


def frobenius_distance(U_target: np.ndarray, U_gen: np.ndarray) -> float:
    """
    Same style as the paper discussion:
        0.5 * ||U_target - U_gen||_F^2
    """
    return 0.5 * np.linalg.norm(U_target - U_gen, ord="fro") ** 2


def rank_circuits_by_distance(
    circuits: list,
    backend: CircuitsQiskitBackend,
    U_target: np.ndarray,
):
    """
    Compute a simple ranking of generated circuits by Frobenius distance
    to the target unitary.
    """
    ranked = []

    for idx, qc in enumerate(circuits):
        if qc is None:
            continue

        try:
            U_gen = backend.get_unitary(qc)
            dist = frobenius_distance(U_target, U_gen)
            ranked.append((idx, dist, qc))
        except Exception:
            continue

    ranked.sort(key=lambda x: x[1])
    return ranked

# POTENTIAL LIBRARY bug WORKAROUND:
def prepare_ddpm_scheduler_for_inference(scheduler):
    """
    Local workaround for a DDPMScheduler bug:
    step(...) expects several derived tensors that are not initialized
    in __init__ but are required during inference.
    """
    scheduler.sqrt_alphas_cumprod = scheduler.alphas_cumprod.sqrt()
    scheduler.sqrt_one_minus_alphas_cumprod = (1.0 - scheduler.alphas_cumprod).sqrt()
    scheduler.sqrt_alphas = scheduler.alphas.sqrt()
    return scheduler

def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------

    # Path where your training script saved the pipeline
    TRAIN_OUT_DIR = Path("runs/dm1_discrete_compile_q4_smoketest_trained")

    # Use the same gate pool as in training
    gate_pool = ["h", "x", "z", "cx", "ccx"]

    # Inference settings
    batch_size = 8
    guidance_scale = 7.5

    # Use fewer inference steps than training-time steps to make sampling faster.
    # DDPMScheduler.set_timesteps(...) requires num_inference_steps < num_train_timesteps.
    num_inference_steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # LOAD PIPELINE
    # ------------------------------------------------------------

    print("\n[STEP 1] Loading trained pipeline ...")
    pipeline = DiffusionPipeline_Compilation.from_config_file(
        config_path=str(TRAIN_OUT_DIR) + "/",
        device=device,
    )

    # Local scheduler compatibility patch for DDPMScheduler inference
    pipeline.scheduler = prepare_ddpm_scheduler_for_inference(pipeline.scheduler)

    # Use a shorter inference schedule for faster sampling
    pipeline.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # ------------------------------------------------------------
    # RECOVER SHAPES / TOKENIZER INFO
    # ------------------------------------------------------------

    # We stored dataset metadata inside pipeline.add_config during training.
    dataset_params = pipeline.add_config["dataset"]["params"]

    num_qubits = int(dataset_params["num_of_qubits"])
    max_gates = int(dataset_params["max_gates"])

    # The embedder works in a continuous latent space with clr_dim channels.
    clr_dim = int(pipeline.embedder.clr_dim)

    tokenizer = CircuitTokenizer({gate: i + 1 for i, gate in enumerate(gate_pool)})
    backend = CircuitsQiskitBackend()

    print(f"Loaded pipeline on device         : {device}")
    print(f"Recovered num_qubits              : {num_qubits}")
    print(f"Recovered max_gates               : {max_gates}")
    print(f"Recovered embedder clr_dim        : {clr_dim}")
    print(f"Inference steps                   : {num_inference_steps}")
    print(f"Batch size                        : {batch_size}")

    # ------------------------------------------------------------
    # BUILD TARGET UNITARY
    # ------------------------------------------------------------

    print("\n[STEP 2] Building target unitary from an example circuit ...")
    target_qc = build_example_target_circuit(num_qubits=num_qubits)
    U_target = backend.get_unitary(target_qc)
    U_cond = unitary_to_condition_tensor(U_target, batch_size=batch_size, device=device)

    print("Target circuit:")
    print(target_qc)
    print(f"Target unitary shape              : {U_target.shape}")

    # ------------------------------------------------------------
    # BUILD TEXT CONDITION
    # ------------------------------------------------------------

    print("\n[STEP 3] Building text condition ...")
    c = build_prompt_tokens(
        pipeline=pipeline,
        gate_pool=gate_pool,
        batch_size=batch_size,
        device=device,
    )
    print(f"Prompt token tensor shape         : {tuple(c.shape)}")

    # ------------------------------------------------------------
    # SAMPLE RANDOM LATENT NOISE
    # ------------------------------------------------------------

    print("\n[STEP 4] Sampling latent noise ...")
    latents = torch.randn(
        (batch_size, clr_dim, num_qubits, max_gates),
        device=device,
        dtype=torch.float32,
    )
    print(f"Latent noise shape                : {tuple(latents.shape)}")

    # ------------------------------------------------------------
    # RUN DIFFUSION INFERENCE
    # ------------------------------------------------------------

    print("\n[STEP 5] Running diffusion inference ...")
    latents_pred = pipeline(
        latents=latents,
        c=c,
        U=U_cond,
        g=guidance_scale,
        no_bar=False,
    )
    print(f"Predicted latent shape            : {tuple(latents_pred.shape)}")

    # ------------------------------------------------------------
    # DECODE TO CIRCUITS
    # ------------------------------------------------------------

    print("\n[STEP 6] Decoding generated latents back to circuits ...")
    token_tensors, instructions_list, circuits = decode_latents_to_circuits(
        pipeline=pipeline,
        latents_pred=latents_pred,
        tokenizer=tokenizer,
        backend=backend,
    )

    print(f"Decoded token tensor shape        : {tuple(token_tensors.shape)}")
    num_valid = sum(qc is not None for qc in circuits)
    print(f"Valid decoded circuits            : {num_valid} / {len(circuits)}")

    # ------------------------------------------------------------
    # RANK BY UNITARY DISTANCE
    # ------------------------------------------------------------

    print("\n[STEP 7] Ranking candidates by Frobenius distance ...")
    ranked = rank_circuits_by_distance(
        circuits=circuits,
        backend=backend,
        U_target=U_target,
    )

    if len(ranked) == 0:
        print("No valid circuits could be ranked.")
        return

    print("\nTop candidates:")
    for rank, (idx, dist, qc) in enumerate(ranked[:5], start=1):
        print("-" * 80)
        print(f"Rank {rank}")
        print(f"Sample index        : {idx}")
        print(f"Frobenius distance  : {dist:.6f}")
        print(qc)

    # ------------------------------------------------------------
    # SAVE BEST CIRCUIT TEXTUALLY
    # ------------------------------------------------------------

    print("\n[STEP 8] Saving best candidate ...")
    best_idx, best_dist, best_qc = ranked[0]

    out_dir = TRAIN_OUT_DIR / "inference_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_candidate.txt", "w") as f:
        f.write(f"Best sample index: {best_idx}\n")
        f.write(f"Frobenius distance: {best_dist:.6f}\n\n")
        f.write(str(best_qc))

    with open(out_dir / "target_circuit.txt", "w") as f:
        f.write(str(target_qc))

    print(f"Saved best candidate to: {out_dir / 'best_candidate.txt'}")
    print(f"Saved target circuit to: {out_dir / 'target_circuit.txt'}")


if __name__ == "__main__":
    main()