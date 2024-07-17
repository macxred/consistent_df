# consistent_df

[![codecov](https://codecov.io/gh/macxred/consistent_df/branch/main/graph/badge.svg)](https://codecov.io/gh/macxred/consistent_df)

`consistent_df` is a powerful Python package that streamlines DataFrame operations across multiple modules.
It ensures type consistency, leverages pandas-like drop indexes, and simplifies DataFrame manipulations.
By centralizing these operations, `consistent_df` boosts efficiency and maintains high-quality code, ensuring
robust testing and consistency across various projects.

## Key Features:
- Data type enforcement for DataFrames.
- Nesting and unnesting operations for hierarchical data.
- String manipulation functions for DataFrames.
- Validation utilities for testing DataFrame content.

## Installation

Easily install the package using pip:

```bash
pip install https://github.com/macxred/consistent_df_df/tarball/main
```

## Testing Strategy

Tests are housed in the [consistent_df_df/tests](tests) directory and are automatically executed via GitHub Actions. This ensures that the code is tested after each commit, during pull requests, and on a daily schedule. We prefer pytest for its straightforward and readable syntax over the unittest package from the standard library.

## Package Development and Contribution

See [cashctrl_api /CONTRIBUTING.md](https://github.com/macxred/cashctrl_api/blob/main/CONTRIBUTING.md) for:

- Setting Up Your Development Environment
- Type Consistency with DataFrames
- Standards and Best Practices
- Leveraging AI Tools
- Shared Learning through Open Source