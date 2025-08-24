import warnings

warnings.warn(
    "get_with_options.py is deprecated (ad-hoc inspection). Prefer repository queries or examples/.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    print(
        "Deprecated: use repositories/contract_descriptions and pandas queries, or add a small example under examples/."
    )
