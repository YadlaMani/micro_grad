# micrograd notes

I learned how neural nets work from Andrej Karpathy's [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0). I coded along and wrote notes as I went, so I can come back and recall it later. Hope it's helpful to anyone else learning this too.

The code and notes are in `micro_grad.ipynb`.

## What's inside

1. **Derivatives by nudging**: `(f(x+h) - f(x)) / h` measures how much an input moves the output.
2. **`Value` class**: wraps a number and records the values and operation that produced it, building a computation graph.
3. **Graph drawing**: a graphviz view of the computation, with data and grad on every node.
4. **Backpropagation**: the chain rule run from output to inputs. Each op knows its local derivative and multiplies it by the gradient coming from above. `backward()` topologically sorts the graph and walks it in reverse.
5. **Neuron with `tanh`**: built by hand, then checked against PyTorch.
6. **Neuron → Layer → MLP**: weighted sum + bias → activation, stacked into layers.
7. **Loss and training**: squared-error loss, `loss.backward()`, then `p.data += -lr * p.grad` in a loop.

## Run it

```bash
pip install numpy matplotlib graphviz torch jupyter
# graphviz also needs the system package: brew install graphviz / sudo apt install graphviz
jupyter notebook micro_grad.ipynb
```

## Things to remember

- **`+=` in backward**: a value used in two places collects gradient from both.
- **Zero the grads before `loss.backward()`**: set `p.grad = 0.0` for every parameter on each training step. Because `_backward` uses `+=`, grads from earlier steps otherwise pile up and every update uses the wrong, ever-growing gradient. This is the same job as `optimizer.zero_grad()` in PyTorch.
- **The minus in the update**: the gradient points uphill on the loss, so step the other way.
- **`tanh` saturation**: outputs live in (-1, 1). Targets outside that range (like `[2, -5, -1, 4]` in the training cell) push outputs to ±1, gradients go to ~0, and the loss freezes at 26.0. Keep targets in range, e.g. `[1, -1, -1, 1]`.

## References

- Video: https://www.youtube.com/watch?v=VMj-3S1tku0
- micrograd repo: https://github.com/karpathy/micrograd
- CS231n neuron notes: https://cs231n.github.io/neural-networks-1/
# micro_grad
