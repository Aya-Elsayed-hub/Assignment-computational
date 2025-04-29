import numpy as np
 
text = "I love deep learning"
words = text.split()
vocab = list(set(words))
vocab_size = len(vocab)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

X = [word2idx[w] for w in words[:3]]   
Y = word2idx[words[3]]                

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

X_oh = np.array([one_hot(i, vocab_size) for i in X])   
 
np.random.seed(42)
hidden_size = 8

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

 
def forward(inputs):
    h = np.zeros((hidden_size, 1))
    for x in inputs:
        x = x.reshape(-1, 1)
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = softmax(y)
    return p, h
 
learning_rate = 0.1
target_vec = one_hot(Y, vocab_size).reshape(-1, 1)

for epoch in range(500):
    p, h = forward(X_oh)

    loss = -np.sum(target_vec * np.log(p))

    dy = p - target_vec
    dWhy = np.dot(dy, h.T)
    dby = dy

    dh = np.dot(Why.T, dy) * (1 - h ** 2)
    dWxh = np.dot(dh, X_oh[-1].reshape(1, -1))
    dWhh = np.dot(dh, h.T)
    dbh = dh

    for param, dparam in zip([Wxh, Whh, Why, bh, by], 
                             [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam

    if (epoch + 1) % 100 == 0:
        pred_idx = np.argmax(p)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Prediction: {idx2word[pred_idx]}")

 
p, _ = forward(X_oh)
predicted_word = idx2word[np.argmax(p)]
print("Final Prediction:", predicted_word)
