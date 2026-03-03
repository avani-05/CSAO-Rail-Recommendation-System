This folder represents the initial raw event-level dataset used to build
the cart-based recommendation system.

Tables:

- users.csv → user-level metadata
- sessions.csv → session boundaries and timestamps
- restaurants.csv → restaurant metadata
- items_noisy.csv → item-level metadata (with injected noise)
- cart_events.csv → sequential cart interaction events

Full dataset statistics:
- 10,000 users
- 94,417 sessions
- 195,579 cart events
- 5,575 items

Only small samples are included in this repository for reproducibility.
