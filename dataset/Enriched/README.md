This folder contains transformed datasets derived from the raw event logs.

Stage 1 – LLM-based Item Enrichment:
- items_noisy.csv cleaned using Gemini API
- Standardized item names
- Extracted cuisine tags
- Removed menu noise and formatting inconsistencies
- Generated structured semantic tags

Stage 2 – Feature Enrichment:
- Time-of-day bucket encoding
- Restaurant-level popularity features
- Item-level frequency features
- Cart context aggregation

