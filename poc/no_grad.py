import torch

x = torch.tensor([1.0], requires_grad=True)
print(f"1. {x.requires_grad=}")

with torch.no_grad():
    y = x * 2
    print(f"2. {y.requires_grad=}")

print(f"3. {y.requires_grad=}")


@torch.no_grad()
def f():
    x = torch.tensor([1.0], requires_grad=True)
    print(f"4. {x.requires_grad=}")
    y = x * 2
    print(f"5. {y.requires_grad=}")
    g(x, y)


def g(x, y):
    print(f"6. {x.requires_grad=}")
    print(f"7. {y.requires_grad=}")
    w = x * y
    print(f"8. {w.requires_grad=}")
    z = x * x
    print(f"9. {z.requires_grad=}")


f()
