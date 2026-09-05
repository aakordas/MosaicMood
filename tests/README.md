# Testing

Run full tests with coverage:

```sh
python3.12 -m pytest -q --cov=. --cov-report=term-missing
```

Run only unit or integration tests:

```sh
python3.12 -m pytest -q -m unit
python3.12 -m pytest -q -m integration
```
