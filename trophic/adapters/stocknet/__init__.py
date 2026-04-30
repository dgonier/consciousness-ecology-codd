"""StockNet (ACL-18) dataset adapters.

Two concrete `Adapter` subclasses for the StockNet preprocessed cache:

- `OhlcvNormalizedAdapter` -> source="ohlcv"
- `TokenizedTweetAdapter`  -> source="tweets"

Both are pure data transforms; live HTTP fetching lives in
`trophic/training/stocknet_loader.py` and is the integration owner's concern.
"""
from .ohlcv_normalized import OhlcvNormalizedAdapter
from .tokenized_tweet import TokenizedTweetAdapter

__all__ = ["OhlcvNormalizedAdapter", "TokenizedTweetAdapter"]
