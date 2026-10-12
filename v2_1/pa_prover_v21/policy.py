"""Fresh autoregressive macro policy with a budget-conditioned value head.

Text lemma memory is the baseline. No recursive lemma encoder or latent loop
is used. A shared prompt KV cache is reused across candidate continuations.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pa_prover_v2.model import RMSNorm, RotaryEmbedding, SwiGLU
from pa_prover_v2.tokenizer import ByteTokenizer

FORMAT="pa-prover-v21-macro-policy-1"


@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    max_seq_len: int = 8192
    ffn_multiplier: float = 8/3
    exploration: float = .05

    def __post_init__(self):
        if min(self.d_model,self.n_heads,self.n_layers,self.max_seq_len)<=0 or self.d_model%self.n_heads or (self.d_model//self.n_heads)%2:
            raise ValueError("invalid Transformer dimensions")
        if not 0<=self.exploration<1:raise ValueError("exploration must be in [0,1)")


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.n_heads;self.width=c.d_model//c.n_heads
        self.qkv=nn.Linear(c.d_model,3*c.d_model,bias=False)
        self.out=nn.Linear(c.d_model,c.d_model,bias=False);self.rope=RotaryEmbedding(self.width)

    def forward(self,h,cache=None):
        batch,length,width=h.shape;offset=0 if cache is None else cache[0].shape[2]
        q,k,v=self.qkv(h).view(batch,length,3,self.heads,self.width).unbind(2)
        pos=torch.arange(offset,offset+length,device=h.device)[None].expand(batch,-1)
        q=self.rope(q.transpose(1,2),pos);k=self.rope(k.transpose(1,2),pos);v=v.transpose(1,2)
        if cache is not None:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
        mask=torch.arange(k.shape[2],device=h.device)[None,:]<=torch.arange(offset,offset+length,device=h.device)[:,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.out(y.transpose(1,2).contiguous().view(batch,length,width)),(k,v)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=RMSNorm(c.d_model);self.n2=RMSNorm(c.d_model)
        self.attn=Attention(c);self.ffn=SwiGLU(c)
    def forward(self,h,cache=None):
        y,cache=self.attn(self.n1(h),cache);h=h+y
        return h+self.ffn(self.n2(h)),cache


def prompt(observation):
    return "STATE\n"+json.dumps(observation,sort_keys=True,separators=(",",":"))+"\nACTION\n"


class MacroPolicy(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or PolicyConfig();c=self.config
        self.tokenizer=ByteTokenizer();self.embedding=nn.Embedding(260,c.d_model)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.n_layers));self.norm=RMSNorm(c.d_model)
        self.head=nn.Linear(c.d_model,260,bias=False);self.value=nn.Linear(c.d_model,1)
        self.provenance={"initialization":"random","human_theorem_library":False}

    @property
    def device(self):return self.embedding.weight.device

    def forward(self,ids,cache=None):
        offset=0 if cache is None else cache[0][0].shape[2]
        if ids.shape[1]+offset>self.config.max_seq_len:raise ValueError("complete state exceeds context; truncation is forbidden")
        h=self.embedding(ids);new=[]
        for i,block in enumerate(self.blocks):h,c=block(h,None if cache is None else cache[i]);new.append(c)
        h=self.norm(h)
        return self.head(h),self.value(h[:,-1]).squeeze(-1),new

    def ids(self,text,*,bos=False,eos=False):
        return torch.tensor([self.tokenizer.encode(text,bos=bos,eos=eos)],device=self.device)

    def sequence_loss(self,text,prefix=""):
        ids=self.ids(prefix+text,bos=True,eos=True)
        logits,_,_=self(ids[:,:-1]);labels=ids[:,1:].clone()
        if prefix:labels[:,:len(self.tokenizer.encode(prefix))]=-100
        return F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)

    def distribution(self,observation,actions,temperature=1.,*,grad=False):
        if not actions or temperature<=0:raise ValueError("nonempty candidates and positive temperature required")
        if len(set(actions))!=len(actions):raise ValueError("duplicate action candidates")
        with torch.set_grad_enabled(grad):
            logits,value,cache=self(self.ids(prompt(observation),bos=True))
            scores=[]
            for a in actions:
                ids=self.ids(a,eos=True)
                continuation,_,_=self(ids[:,:-1],cache)
                predictions=torch.cat((logits[:,-1:],continuation),dim=1)
                lp=F.log_softmax(predictions.float(),-1).gather(-1,ids.unsqueeze(-1)).sum()
                scores.append(lp)
            logps=F.log_softmax(torch.stack(scores)/temperature,0)
            if self.config.exploration:
                logps=torch.logaddexp(logps+math.log1p(-self.config.exploration),
                                     logps.new_tensor(math.log(self.config.exploration/len(actions))))
            return logps,value[0].sigmoid()

    def save(self,path,**metadata):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save({"format":FORMAT,"config":asdict(self.config),"tokenizer":self.tokenizer.fingerprint(),
                    "state":self.state_dict(),"provenance":self.provenance,"metadata":metadata},path)

    @classmethod
    def load(cls,path,device="cpu"):
        r=torch.load(path,map_location="cpu",weights_only=True)
        if r.get("format")!=FORMAT or r.get("tokenizer")!=ByteTokenizer().fingerprint():
            raise ValueError("checkpoint architecture/tokenizer is not a v2.1 macro policy")
        if r.get("provenance",{}).get("initialization")!="random":raise ValueError("strict v2.1 requires random initialization provenance")
        model=cls(PolicyConfig(**r["config"]));model.load_state_dict(r["state"],strict=True)
        model.provenance=r["provenance"];model.eval();return model.to(device)


def multi_positive_loss(log_probabilities,positive_indices):
    positive_indices=sorted(set(positive_indices))
    if not positive_indices or any(i<0 or i>=len(log_probabilities) for i in positive_indices):
        raise ValueError("positive actions must belong to the sampled candidate set")
    return -torch.logsumexp(log_probabilities[positive_indices],0)


def ppo_loss(new_logps,old_logps,index,advantage,reference_logps,clip=.2,kl_weight=.02):
    """All probabilities refer to the identical finite candidate distribution."""
    old_logps=old_logps.detach();reference_logps=reference_logps.detach()
    ratio=(new_logps[index]-old_logps[index]).exp()
    clipped=ratio.clamp(1-clip,1+clip)
    objective=-torch.minimum(ratio*advantage,clipped*advantage)
    kl=(new_logps.exp()*(new_logps-reference_logps)).sum()
    return objective+kl_weight*kl
