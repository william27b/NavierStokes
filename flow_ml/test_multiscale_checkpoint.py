"""Check restart correctness and separation of inference inputs from labels."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from multiscale_graph import build_graph
from multiscale_model import MultiscaleGNN, FlowLoss
from train_multiscale import save_checkpoint, load_new

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        solid=np.zeros((8,8),bool)
        solid[[0,-1]]=True
        self.graph=build_graph(solid,levels=2)
        self.graph['y']=torch.randn(len(self.graph['x']),3)*.02

    def test_optimizer_scheduler_restart_matches_next_update(self):
        config=dict(hidden=8,levels=2,local_steps=1,coarse_steps=2)
        model=MultiscaleGNN(**config)
        loss_fn=FlowLoss([.1,.2,.3],[.02,.03,.04])
        opt=torch.optim.AdamW(model.parameters(),lr=.0003)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
        rng=torch.Generator().manual_seed(9)
        def step(m,o,s):
            o.zero_grad(set_to_none=True)
            loss,_=loss_fn(m(self.graph),self.graph)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            o.step();s.step()
        step(model,opt,schedule)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt'
            save_checkpoint(path,model,opt,schedule,1,loss_fn,rng,{})
            restored, restored_loss, memory=load_new(path,'cpu')
            opt2=torch.optim.AdamW(restored.parameters(),lr=.9)
            opt2.load_state_dict(memory['optimizer_state'])
            schedule2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=99)
            schedule2.load_state_dict(memory['scheduler_state'])
            torch.testing.assert_close(model(self.graph),restored(self.graph),rtol=0,atol=0)
            step(model,opt,schedule);step(restored,opt2,schedule2)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(schedule.get_last_lr(),schedule2.get_last_lr())
            for a,b in zip(opt.state.values(),opt2.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            self.assertEqual(restored_loss.gradient_weight,loss_fn.gradient_weight)

    def test_predictions_do_not_read_labels(self):
        model=MultiscaleGNN(hidden=8,levels=2,local_steps=1,coarse_steps=2).eval()
        with torch.no_grad():
            original=model(self.graph)
            self.graph['y'].fill_(float('nan'))
            corrupted=model(self.graph)
            del self.graph['y']
            geometry_only=model(self.graph)
        torch.testing.assert_close(original,corrupted,rtol=0,atol=0)
        torch.testing.assert_close(original,geometry_only,rtol=0,atol=0)

if __name__=='__main__':
    unittest.main(verbosity=2)
