import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch
import wandb
import json
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_classes, embed_size=30):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.embedding_layer = nn.Linear(1000, embed_size)
        self.output_layer = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x2 = self.resnet(x).detach()
        #x2 = self.resnet(x)
        embedded = self.embedding_layer(x2)
        class_logits = self.output_layer(embedded)

        return class_logits, embedded

def validate(model, val_loader, e=-1):
    model.eval()
    with torch.no_grad():
        val_loss, total_correct, total = 0, 0, 0
        Y_pred, Y_true, names = [], [], []
        for _, samp in enumerate(val_loader):
            Y, X, N = samp
            logits, emb = model(X)
            val_loss += F.cross_entropy(logits, Y, reduction="sum").item()
            Y_pred.extend(torch.argmax(logits, dim=1))
            Y_true.append(Y)
            names.extend(N)
            total += len(X) 
            #Y_true = torch.argmax(Y, dim=1)

        Y_true = torch.concatenate(Y_true)
        #print(Y_true.shape)
        for i, pr in enumerate(Y_pred):
            if(Y_true[i, pr] == 1):
                #print("correct", names[i], pr, Y_true[i,:])
                total_correct += 1
                #print("failure", names[i], pr, Y_true[i,:])
            #total_correct += sum(Y_pred==Y).item()
            
        #print(f"")
        #print(f"E:{e+1} Validation Loss:{val_loss / total:.2f}")
        print(f"E:{e+1} acc:{total_correct / total:.2%} loss:{val_loss / total:.2f}")
        if(e != -1): wandb.log({"epoch": e+1, "validation_accuracy":total_correct / total, "validation_loss": val_loss / total})

def train_model(model, train_loader, val_loader, name, config={}, num_epochs=15, lr=1e-3, focal_loss=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if(focal_loss):
        gamma = 2.0
        alpha = 1.0
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    else:
        criterion = F.cross_entropy

    config["learning_rate"] = lr
    config["focal_loss"] = focal_loss
    wandb.init(project="p_mapping", name=name, config=config)

    for e in range(num_epochs):
        total_loss, total = 0, 0
        model.train()
        for i, samp in enumerate(train_loader):
            Y, X, N = samp

            logits, emb = model(X)

            loss = criterion(logits, Y)
            total_loss += loss.item() * len(X)
            total += len(X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\rsamples:{i+1}/{len(train_loader)} epoch:{e+1}/{num_epochs} loss:{total_loss/total:.2f}  ", end="")

            wandb.log({"epoch": e+i/len(train_loader), "training_loss": loss})
        print()

        validate(model, val_loader, e)
        torch.save(model.state_dict(), f"./saves/{name}_e-{e+1}")


def viewer(model, test_loader):
    model.eval()
    with torch.no_grad():
        for i, samp in enumerate(test_loader):
            Y, X, N = samp
            logits, emb = model(X)
            yield N, logits, Y


from typing import Optional
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = F.one_hot(target, num_classes=input.shape[1]).to(torch.float32)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path, zoom=0.125):
    return OffsetImage(plt.imread(path), zoom=zoom)

import time
def save_distances(model, data_loader, name2id, id2type, le):
    embeddings = []
    names = []
    nodes = []
    for i, samp in enumerate(data_loader):
        Y, X, N = samp
        logits, emb = model(X)
        #time.sleep(0.1)
        print(f"{i+1}/{len(data_loader)}")
        names.extend(N)
        embeddings.append(emb)
        type1 = id2type[name2id[N[0]]]
        nodes.append({"id":N[0], "group":str(le.transform([type1])[0])})
        #if(i == 20): break
    embeddings = torch.concatenate(embeddings).detach().numpy()
    from sklearn.manifold import TSNE
    embeddings =  TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)
    embeddings = torch.Tensor(embeddings)
    ids = [name2id[x] for x in names]  
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    links = []
    for i, embi in enumerate(embeddings):
        topk = []
        for j, embj in enumerate(embeddings):
            if(i == j): continue
            topk.append((j, torch.cdist(embi.unsqueeze(0), embj.unsqueeze(0)).item()))
        topk.sort(key=lambda x: x[1])
        for j, val in topk[-3:]:
            links.append({"source":names[i],"target":names[j],"value":val})
    max_link_val = max([x["value"] for x in links])
    min_link_val = min([x["value"] for x in links])
    for i in range(len(links)):
        links[i]["value"] = (links[i]["value"] - min_link_val) / (max_link_val - min_link_val)

    json_object = json.dumps({"nodes":nodes, "links":links}, indent=4)
    with open("./graph.json", "w+") as outfile:
        outfile.write(json_object)

def visualize(model, data_loader, name2id):
    embeddings = []
    names = []
    for i, samp in enumerate(data_loader):
        Y, X, N = samp
        logits, emb = model(X)
        print(f"{i+1}/{len(data_loader)}")
        names.extend(N)
        embeddings.append(emb)
        break
    ids = [name2id[x] for x in names]
    print(names)

    #embeddings.append(emb)
    embeddings = torch.concatenate(embeddings).detach().numpy()
    print(embeddings.shape)

    layout = fl.draw_spring_layout(dataset=embeddings, algorithm=fl.SpringForce)
    from sklearn.manifold import TSNE
    mapped =  TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)
    print(mapped.shape)

    plt.figure(figsize=(80, 80), dpi=160)
    fig, ax = plt.subplots()
    ax.scatter(mapped[:,0], mapped[:,1]) 
    paths = [f'./images/pokemon/{x}.png' for x in ids]

    for x0, y0, path in zip(mapped[:,0], mapped[:,1], paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.savefig("./map.png")