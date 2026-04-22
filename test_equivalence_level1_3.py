import torch
import torch.nn as nn

from genQC.pipeline.equivalence_diffusion_pipeline import DiffusionPipeline_Equivalence
from genQC.models.source_circuit_encoder import SourceCircuitEncoder
from genQC.models.unet_qc import QC_Cond_UNet


# ------------------------------------------------------------
# Small helper modules for a clean smoke test
# ------------------------------------------------------------

class DummyTextEncoder(nn.Module):
    """
    Minimal stand-in for the real text encoder.
    It provides:
      - forward(y, pool=False) -> [b, seq, cond_emb_size]
      - empty_token
      - cached_empty_token_index
      - tokenize_and_push_to_device(...)
    so the pipeline can be tested without depending on OpenCLIP setup.
    """
    def __init__(self, vocab_size=128, seq_len=8, cond_emb_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.cond_emb_size = cond_emb_size

        self.embedding = nn.Embedding(vocab_size, cond_emb_size)

        # attributes used by DiffusionPipeline
        self.empty_token = torch.zeros((1, seq_len), dtype=torch.long)
        self.cached_empty_token_index = 0

    def tokenize_and_push_to_device(self, prompts):
        out = torch.zeros((len(prompts), self.seq_len), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids = [(ord(ch) % self.vocab_size) for ch in p[: self.seq_len]]
            if len(ids) > 0:
                out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def forward(self, y, pool=False):
        if y.dim() == 1:
            y = y.unsqueeze(0)
        y = y.long()
        emb = self.embedding(y)  # [b, seq, cond_emb_size]
        if pool:
            return emb.mean(dim=1)
        return emb


class ModelAsEmbedder(nn.Module):
    """
    Lightweight wrapper so we can use QC_Cond_UNet's embed/invert functions
    without registering the full model twice in self.trainables.
    """
    def __init__(self, model):
        super().__init__()
        self.model_ref = model
        self.channel_last = False

    def embed(self, x):
        return self.model_ref.embed(x)

    def invert(self, x, *args, **kwargs):
        return self.model_ref.invert(x)

    def parameters(self, recurse=True):
        # do not expose parameters here; the actual model is already in trainables
        return iter(())

    def train(self, mode=True):
        # no-op; the real model is handled separately
        return self

    def to(self, device):
        return self


class DummyScheduler:
    """
    Minimal scheduler with the interface needed for train_step(...) up to level 3.
    This verifies the extension wiring without depending on the exact DDPMScheduler constructor.
    """
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps

    def to(self, device):
        return self

    def add_noise(self, latents, noise, timesteps, train=True):
        # simple stable alpha schedule just for smoke testing
        alpha = 1.0 - 0.5 * (timesteps.float() + 1.0) / (self.num_train_timesteps + 1.0)
        alpha = alpha.clamp(min=1e-4, max=0.999).view(-1, 1, 1, 1).to(latents.device)
        return alpha.sqrt() * latents + (1.0 - alpha).sqrt() * noise


def any_param_changed(before, after, atol=1e-12):
    """
    Returns:
        changed: bool
        max_diff: float
    """
    max_diffs = [(a - b).abs().max().item() for a, b in zip(after, before)]
    return any(d > atol for d in max_diffs), max(max_diffs) if len(max_diffs) > 0 else 0.0


# ------------------------------------------------------------
# Build the new equivalence-training stack
# ------------------------------------------------------------

torch.manual_seed(0)
device = torch.device("cpu")

# keep this small so the smoke test runs fast
cond_emb_size = 64
num_clrs = 8
num_qubits = 4
max_gates = 8
batch_size = 4

model = QC_Cond_UNet(
    model_features=[32, 32, 64],
    clr_dim=8,
    num_clrs=num_clrs,
    t_emb_size=64,
    cond_emb_size=cond_emb_size,
    num_heads=[4, 4, 4],
    num_res_blocks=[1, 1, 1],
    transformer_depths=[1, 1, 1],
)

text_encoder = DummyTextEncoder(
    vocab_size=128,
    seq_len=8,
    cond_emb_size=cond_emb_size,
)

embedder = ModelAsEmbedder(model)

scheduler = DummyScheduler(num_train_timesteps=1000)

source_encoder = SourceCircuitEncoder(
    in_channels=8,
    cond_emb_size=cond_emb_size,
    hidden_channels=32,
    num_heads=4,
    depth=2,
    mlp_ratio=2,
    dropout=0.0,
    add_positional_encoding=True,
)

pipeline = DiffusionPipeline_Equivalence(
    scheduler=scheduler,
    model=model,
    text_encoder=text_encoder,
    embedder=embedder,
    source_encoder=source_encoder,
    device=device,
    enable_guidance_train=False,   # disable CFG for the first smoke test
    guidance_train_p=0.1,
    cached_text_enc=True,
)

# ------------------------------------------------------------
# Build a synthetic batch
# ------------------------------------------------------------

target_tokens = torch.randint(
    low=0,
    high=num_clrs,
    size=(batch_size, num_qubits, max_gates),
    dtype=torch.long,
)

source_tokens = torch.randint(
    low=0,
    high=num_clrs,
    size=(batch_size, num_qubits, max_gates),
    dtype=torch.long,
)

prompts = ["Generate equivalent circuit using: ['h','cx','x']"] * batch_size
y_tokens = text_encoder.tokenize_and_push_to_device(prompts)


# ------------------------------------------------------------
# LEVEL 1: imports + construction + summary + trainables registration
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEVEL 1: construction / summary / trainables registration")
print("=" * 80)

print(pipeline.summary())

assert hasattr(pipeline, "source_encoder"), "Pipeline has no source_encoder attribute."
assert any(obj is source_encoder for obj in pipeline.trainables), \
    "source_encoder is NOT registered in pipeline.trainables."

print("[OK] source_encoder is registered in pipeline.trainables")


# ------------------------------------------------------------
# LEVEL 1.5: compile(...)
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEVEL 1.5: compile(...)")
print("=" * 80)

pipeline.compile(
    optim_fn=torch.optim.Adam,
    loss_fn=torch.nn.MSELoss,
    metrics=[],
    lr=1e-3,
)

print("[OK] pipeline.compile(...) completed")


# ------------------------------------------------------------
# LEVEL 2: direct train_step(...)
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEVEL 2: direct train_step(...)")
print("=" * 80)

loss = pipeline.train_step((target_tokens, y_tokens, source_tokens), train=True)

print("train_step loss =", float(loss.detach()))
assert torch.isfinite(loss).item(), "train_step returned a non-finite loss."

print("[OK] train_step(...) returned a finite scalar loss")


# ------------------------------------------------------------
# LEVEL 3: one train_on_batch(...)
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEVEL 3: one train_on_batch(...)")
print("=" * 80)

# snapshot params before update
model_params_before = [p.detach().clone() for p in pipeline.model.parameters()]
src_params_before = [p.detach().clone() for p in pipeline.source_encoder.parameters()]

batch_loss = pipeline.train_on_batch((target_tokens, y_tokens, source_tokens), train=True)

print("train_on_batch loss =", float(batch_loss))
assert torch.isfinite(batch_loss).item(), "train_on_batch returned a non-finite loss."

# inspect gradient presence
model_grad_count = sum(int(p.grad is not None) for p in pipeline.model.parameters())
src_grad_count = sum(int(p.grad is not None) for p in pipeline.source_encoder.parameters())

print("model params with grad            =", model_grad_count)
print("source_encoder params with grad   =", src_grad_count)

assert model_grad_count > 0, "No gradients found on the main denoiser model."
assert src_grad_count > 0, "No gradients found on source_encoder."

# inspect parameter change across all params
model_params_after = [p.detach() for p in pipeline.model.parameters()]
src_params_after = [p.detach() for p in pipeline.source_encoder.parameters()]

model_changed, model_max_diff = any_param_changed(model_params_before, model_params_after)
src_changed, src_max_diff = any_param_changed(src_params_before, src_params_after)

print("model max abs param diff          =", model_max_diff)
print("source_encoder max abs param diff =", src_max_diff)
print("any model param changed           =", model_changed)
print("any source_encoder param changed  =", src_changed)

assert model_changed, "No model parameter changed after optimizer step."
assert src_changed, "No source_encoder parameter changed after optimizer step."

print("[OK] train_on_batch(...) worked, gradients flowed, and parameters updated.")

print("\nAll checks up to LEVEL 3 passed.")