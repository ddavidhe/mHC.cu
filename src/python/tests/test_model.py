import pytest
import torch

from mhc.model import ModelConfig, MHCTransformer


def _small_config(**overrides):
    defaults = dict(
        vocab_size=128,
        n_layers=2,
        hidden_dim=64,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=16,
        expansion_rate=2,
        sinkhorn_iters=2,
        rmsnorm_eps=1e-5,
        sinkhorn_eps=1e-5,
        use_fused_mhc=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_model_forward_backward_cpu():
    config = _small_config()
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_swiglu_mlp_cpu():
    config = _small_config(mlp_type="swiglu")
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_gelu_mlp_cpu():
    config = _small_config(mlp_type="gelu")
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_static_h_cpu():
    config = _small_config(use_dynamic_h=False)
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_recompute_mhc_cpu():
    config = _small_config(recompute_mhc=True)
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_checkpoint_ratio_cpu():
    config = _small_config(recompute_ratio=0.5)
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_return_hidden():
    config = _small_config()
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    hidden = model(input_ids, return_hidden=True)
    assert hidden.shape == (2, 8, config.hidden_dim)


def test_model_forward_backward_fused_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused mHC")
    try:
        import mhc_cuda  # noqa: F401
    except ImportError:
        pytest.skip("mhc_cuda extension not available")
    config = _small_config(n_layers=1, use_fused_mhc=True)
    model = MHCTransformer(config).cuda()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    logits = model(input_ids)
    assert logits.is_cuda
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None
