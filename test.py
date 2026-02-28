import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import iresnet, get_vit
from data.load_test_sets_recognition import get_val_pair
import verification
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

class Test:
    def __init__(self, args, local_rank):
        self.device = torch.device(f"cuda:{local_rank}")
        self.local_rank = local_rank
        self.args = args
        
        model = self.create_model(args)
        model = model.to(self.device)
        model.eval()
        
        self.model = DDP(model, device_ids=[local_rank])
            
        self.add_flip = args.add_flip
        self.add_norm = args.add_norm
        self.cosine = args.cosine

        self.validation_list = []
        for val_name in args.val_list:
            dataset, issame = get_val_pair(args.val_source, val_name)
            self.validation_list.append([dataset, issame, val_name])

    def create_model(self, args):
        if args.model == "iresnet":
            model = iresnet(args.depth, fp16=True, mode=args.mode)
        elif args.model == "vit":
            model = get_vit(args.depth)
        
        state_dict = torch.load(args.model_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        return model

    def l2_norm(self, input: torch.Tensor, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output, norm

    def evaluate(self):
        if self.local_rank == 0:
            print("Validating (DDP Mode)...")
        
        val_acc = 0
        for idx, validation in enumerate(self.validation_list):
            dataset, issame, val_name = validation
            acc, std = self.evaluate_recognition(dataset, issame)
            
            if self.local_rank == 0:
                print(f"{val_name}: {acc:.5f}+-{std:.5f}")
                val_acc += acc
        
        if self.local_rank == 0:
            mean_acc = val_acc / len(self.validation_list)
            print(f"Mean accuracy: {mean_acc:.5f}\n")

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        embedding_length = len(samples) // 2
        world_size = dist.get_world_size()
        
        indices = list(range(embedding_length))
        local_indices = indices[self.local_rank::world_size]
        
        local_embeddings = []

        with torch.no_grad():
            for i in range(0, len(local_indices), self.args.batch_size):
                batch_idx = local_indices[i : i + self.args.batch_size]
                
                batch_or = torch.tensor(samples[batch_idx]).to(self.device)
                flip_idx = [idx + embedding_length for idx in batch_idx]
                batch_flip = torch.tensor(samples[flip_idx]).to(self.device)
                with torch.cuda.amp.autocast():
                    if self.add_flip:
                        feat = (self.model(batch_or) + self.model(batch_flip))
                    elif self.add_norm:
                        embeddings_flip, norms_flip = self.l2_norm(self.model(batch_flip), axis=1)
                        embeddings_or, norms_or = self.l2_norm(self.model(batch_or), axis=1)
                        feat = (embeddings_flip * norms_flip + embeddings_or * norms_or)
                    else:
                        feat = self.model(batch_or)
                
                local_embeddings.append(feat.cpu().numpy())

        if len(local_embeddings) > 0:
            local_embeddings = np.concatenate(local_embeddings, axis=0)
        else:
            local_embeddings = np.empty((0, 512))

        all_embeddings_list = [None for _ in range(world_size)]
        dist.all_gather_object(all_embeddings_list, local_embeddings)
        
        if self.local_rank == 0:
            final_embeddings = np.zeros([embedding_length, 512])
            for r, emb in enumerate(all_embeddings_list):
                final_embeddings[r::world_size] = emb
            
            normalized_embedding = np.divide(final_embeddings, np.linalg.norm(final_embeddings, 2, 1, True))
            tpr, fpr, accuracy = verification.evaluate(
                normalized_embedding, issame, nrof_folds, cosine=self.cosine
            )
            return accuracy.mean(), accuracy.std()
        
        return 0.0, 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test a recognition model.")
    parser.add_argument("--model_path", "-model_path", help="model path.", type=str)
    parser.add_argument("--model", "-model", help="iresnet/vit.", type=str, default="iresnet")
    parser.add_argument("--mode", "-mode", help="using SE attention [normal/se].", type=str, default="normal")
    parser.add_argument("--depth", "-d", help="layers size: resnet [18, 34, 50, 100, 152, 200] / vit [s, b, l].", default="100", type=str)
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--add_flip", "-aflip", help="Add flipped image features.", action="store_true")
    parser.add_argument("--add_norm", "-anorm", help="Add feature norm.", action="store_true")
    parser.add_argument("--cosine", "-cosine", help="cosine distance.", action="store_true")
    parser.add_argument("--val_list", "-v", help="List of images to validate.", nargs="+", 
                        default=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "hadrian_even", "eclipse_even", "twins_very_easy"])
    parser.add_argument("--val_source", "-vs", help="Path to the val images.", default="./test_sets")

    args = parser.parse_args()
    local_rank = setup_ddp()
    set_seed(42)

    test = Test(args, local_rank)
    test.evaluate()

    cleanup_ddp()