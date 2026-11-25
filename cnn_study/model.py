"""Small two-layer CNN for 64 x 64 grayscale radiographs."""
import numpy as np

FILTERS = (8, 12)

def initialize(seed):
    rng = np.random.default_rng(seed)
    def weight(n, m):
        return (rng.standard_normal((n, m)) * np.sqrt(2/n)).astype(np.float32)
    a, b = FILTERS
    return {"w1": weight(16, a), "b1": np.zeros(a, np.float32),
            "w2": weight(9*a, b), "b2": np.zeros(b, np.float32),
            "w3": weight(49*b, 1), "b3": np.zeros(1, np.float32)}

def forward(images, params, dropout=0., rng=None):
    n = len(images)
    a, b = FILTERS
    # First convolution uses 4 x 4 patches, stride four, with shared weights.
    patches = images.reshape(n,16,4,16,4).transpose(0,1,3,2,4).reshape(-1,16)
    first = np.maximum(0, patches @ params["w1"] + params["b1"]).reshape(n,16,16,a)
    # Gather overlapping 3 x 3 windows; flattening here is just an implementation of convolution.
    windows = np.stack([first[:,i:i+14,j:j+14,:] for i in range(3) for j in range(3)],axis=3)
    windows = windows.reshape(-1,9*a)
    second = np.maximum(0, windows @ params["w2"] + params["b2"]).reshape(n,14,14,b)
    features = second.reshape(n,7,2,7,2,b).mean(axis=(2,4)).reshape(n,-1)
    mask = None
    if dropout:
        mask = (rng.random(features.shape) >= dropout).astype(np.float32)/(1-dropout)
        features = features * mask
    logits = (features @ params["w3"] + params["b3"]).ravel()
    return logits, (patches, first, windows, second, features, mask)

def sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits, -60, 60)))

def gradients(images, labels, params, dropout, rng):
    logits, (patches, first, windows, second, features, mask) = forward(images, params, dropout, rng)
    n = len(labels)
    dz = (sigmoid(logits) - labels).reshape(-1,1)/n
    grad = {"w3": features.T @ dz, "b3": dz.sum(axis=0)}
    pooled = dz @ params["w3"].T
    if mask is not None:
        pooled *= mask
    d2 = pooled.reshape(n,7,7,FILTERS[1]).repeat(2,axis=1).repeat(2,axis=2) * .25
    d2 *= second > 0
    d2 = d2.reshape(-1,FILTERS[1])
    grad["w2"] = windows.T @ d2
    grad["b2"] = d2.sum(axis=0)
    window_grads = (d2 @ params["w2"].T).reshape(n,14,14,3,3,FILTERS[0])
    d1 = np.zeros_like(first)
    for i in range(3):
        for j in range(3):
            d1[:,i:i+14,j:j+14] += window_grads[:,:,:,i,j]
    d1 *= first > 0
    d1 = d1.reshape(-1,FILTERS[0])
    grad["w1"] = patches.T @ d1
    grad["b1"] = d1.sum(axis=0)
    return grad

def predict_logits(images, params):
    return np.concatenate([forward(images[i:i+32], params)[0] for i in range(0,len(images),32)])

def train(x, y, vx, vy, dropout, seed=42, epochs=20):
    params = initialize(seed)
    batches = np.random.default_rng(seed + 1)
    masks = np.random.default_rng(seed + 2)
    first = {k:np.zeros_like(v) for k,v in params.items()}
    second = {k:np.zeros_like(v) for k,v in params.items()}
    best_loss, best_params, step = np.inf, None, 0
    history = []
    for epoch in range(1,epochs+1):
        order = batches.permutation(len(x))
        for start in range(0,len(x),32):
            ids = order[start:start+32]
            grads = gradients(x[ids],y[ids],params,dropout,masks)
            step += 1
            for k,g in grads.items():
                first[k] = .9*first[k] + .1*g
                second[k] = .999*second[k] + .001*g*g
                params[k] -= .001 * (first[k]/(1-.9**step)) / (np.sqrt(second[k]/(1-.999**step)) + 1e-8)
        z = predict_logits(vx,params)
        val_loss = float(np.mean(np.logaddexp(0,z)-vy*z))
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {k:v.copy() for k,v in params.items()}
        history.append({"epoch":epoch,"val_loss":val_loss})
        if epoch % 5 == 0:
            print(f"dropout={dropout}: epoch {epoch}, validation loss {val_loss:.4f}", flush=True)
    return best_params, history
