"""Check that chunking and saved-state continuation preserve Adam training."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from model import GNN
from train import train
from continue_training import restore_training, save_checkpoint


def fixture():
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]])
    return [{"x": x, "edge_index": edges, "edge_attr": x[edges[1]]-x[edges[0]],
             "y": torch.full((4, 3), value)} for value in (.1, .2, .3, .4, .5)]


class ContinuationTests(unittest.TestCase):
    def test_chunking_and_restoration_match_uninterrupted_training(self):
        torch.manual_seed(5)
        full = GNN(hidden=64, layers=1, global_context=True)
        split = GNN(hidden=64, layers=1, global_context=True)
        split.load_state_dict(full.state_dict())
        full_rng = torch.Generator().manual_seed(23)
        split_rng = torch.Generator().manual_seed(23)
        full_loader = DataLoader(fixture(), batch_size=2, shuffle=True, collate_fn=list, generator=full_rng)
        split_loader = DataLoader(fixture(), batch_size=2, shuffle=True, collate_fn=list, generator=split_rng)
        full_optimizer = train(full, full_loader, torch.device('cpu'), epochs=3)
        epochs = []
        split_optimizer = train(split, split_loader, torch.device('cpu'), epochs=1, start_epoch=40,
                                on_epoch_end=lambda epoch, loss: epochs.append(epoch))
        returned = train(split, split_loader, torch.device('cpu'), epochs=1, start_epoch=41,
                         optimizer=split_optimizer, on_epoch_end=lambda epoch, loss: epochs.append(epoch))
        self.assertIs(returned, split_optimizer)
        self.assertEqual(epochs, [41, 42])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'saved.pt'
            save_checkpoint(checkpoint, split, split_optimizer, split_rng, 42)
            model, optimizer, generator, memory, restored = restore_training(checkpoint, torch.device('cpu'))
            self.assertTrue(restored)
            self.assertEqual(memory['completed_epochs'], 42)
            loader = DataLoader(fixture(), batch_size=2, shuffle=True, collate_fn=list, generator=generator)
            train(model, loader, torch.device('cpu'), epochs=1, optimizer=optimizer, start_epoch=42)
            for expected, actual in zip(full.parameters(), model.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            for expected, actual in zip(full_optimizer.state.values(), optimizer.state.values()):
                for key in expected:
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_legacy_checkpoint_keeps_weights_and_starts_fresh_adam(self):
        original = GNN(hidden=64, layers=1, global_context=True)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'legacy.pt'
            torch.save({'model_state': original.state_dict(), 'hidden': 64, 'layers': 1,
                        'global_context': True}, checkpoint)
            model, optimizer, _, _, restored = restore_training(checkpoint, torch.device('cpu'))
            self.assertFalse(restored)
            self.assertFalse(optimizer.state)
            for expected, actual in zip(original.parameters(), model.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
