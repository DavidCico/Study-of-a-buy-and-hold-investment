# Study-of-buy-and-hold-investment
In this repository, a buy-and-hold investment is studied using Python and Monte Carlo approach

```python
from typing import Iterator

def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b
```
