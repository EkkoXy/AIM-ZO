import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

spec = importlib.util.spec_from_file_location("activation_gradient", Path(__file__).parents[1] / "scripts/diagnose_activation_gradient.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class AlignmentTests(unittest.TestCase):
    def test_disjoint_reproducible_pools(self):
        a, b = diagnostic.split_indices(100, 32, 32, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), diagnostic.split_indices(100, 32, 32, 42))
        with self.assertRaises(ValueError):
            diagnostic.split_indices(10, 8, 8, 42)

    def test_known_subspace(self):
        x = torch.tensor([[3., 0., 0.], [0., 2., 0.], [0., 0., 0.]])
        q = diagnostic.activation_basis(x, 2)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(2)))
        self.assertAlmostEqual(diagnostic.coverage(torch.tensor([[1., 1., 0.]]), q), 1.)
        self.assertAlmostEqual(diagnostic.coverage(torch.tensor([[0., 0., 1.]]), q), 0.)
        with self.assertRaises(ValueError):
            diagnostic.activation_basis(x, 3)

    def test_rectangular_ffn_and_random(self):
        q = diagnostic.random_basis(20, 4, 42)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(4), atol=1e-6))
        self.assertTrue(torch.equal(q, diagnostic.random_basis(20, 4, 42)))
        self.assertAlmostEqual(diagnostic.coverage(torch.eye(20), q), 4 / 20, places=6)
        self.assertIsNone(diagnostic.coverage(torch.zeros(7, 20), q))
        with self.assertRaises(ValueError):
            diagnostic.coverage(torch.ones(7, 10), q)

    def test_padding_excluded(self):
        x = torch.arange(24.).reshape(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        actual = diagnostic.valid_activation_tokens(x, mask, 10)
        self.assertTrue(torch.equal(actual, x.reshape(6, 4)[[0, 1, 3]]))
        self.assertEqual(len(diagnostic.valid_activation_tokens(x, mask, 2)), 2)

    def test_shared_attention_bases_require_equal_inputs(self):
        x = torch.diag(torch.tensor([4., 3., 2., 1.]))
        tokens = {"a.q_proj": [x], "a.k_proj": [x.clone()], "a.v_proj": [x.flip(1)]}
        bases, counts, reused = diagnostic.reconstruct_bases(tokens, 2, "cpu")
        self.assertEqual(reused, 1)
        self.assertIs(bases["a.q_proj"], bases["a.k_proj"])
        self.assertIsNot(bases["a.q_proj"], bases["a.v_proj"])
        self.assertEqual(counts["a.q_proj"], 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_gpu_exact_svd_coverage(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        generator = torch.Generator().manual_seed(400)
        for dimension in (80, 320):
            x = torch.randn(64, dimension, generator=generator)
            g = torch.randn(31, dimension, generator=generator)
            cpu = diagnostic.activation_basis(x, 16)
            gpu = diagnostic.activation_basis(x.cuda(), 16)
            self.assertAlmostEqual(diagnostic.coverage(g, cpu), diagnostic.coverage(g.cuda(), gpu), places=5)

    def test_context_isolation(self):
        samples = [SimpleNamespace(data={"context": str(i // 2)}) for i in range(100)]
        a, b = diagnostic.reference_indices(samples, "squad", 10, 10, 42)
        self.assertFalse({samples[i].data["context"] for i in a} & {samples[i].data["context"] for i in b})

    def test_scoring_losses_keep_gradients(self):
        from aimzo.tasks.zoregular import ZORegularTask, ZORegularSample
        from aimzo.tasks.zoregular_scoring import TinyTokenizer

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 8)
                self.projection = torch.nn.Linear(8, 256)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

        backend = SimpleNamespace(model=ToyModel(), tokenizer=TinyTokenizer(), device="cpu")
        cases = [
            ("rte", ZORegularSample({"premise": "It rains", "hypothesis": "It is wet"}, [0, 1], 0)),
            ("squad", ZORegularSample({"context": "A red house", "question": "What color?", "title": "house", "answers": ["red"]}, None, ["red"])),
        ]
        for name, sample in cases:
            task = ZORegularTask(task_name=name, data_root="unused")
            backend.model.zero_grad(set_to_none=True)
            loss = diagnostic.sample_loss(backend, task, sample, 64, [None])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(backend.model.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
