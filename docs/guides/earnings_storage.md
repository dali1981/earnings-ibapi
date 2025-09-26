# Earnings Storage Guide

This document describes how upcoming earnings are persisted and how to avoid accidental data loss.

## Dataset layout

- Root: `data/earnings/`
- Partitioning: Hive-style partitions by `collection_date` and `source`:
  - `collection_date=YYYY-MM-DD/source=nasdaq/earnings.parquet`
  - `collection_date=YYYY-MM-DD/source=yahoo/earnings.parquet`

Each `collection_date` represents one ingestion run. Multiple runs over days accumulate, enabling historical comparisons and reprocessing.

## Writing data (safe path)

- API and jobs must use `EarningsCalendarRepository.store_earnings_batch(events, collection_date)`.
- Behavior:
  - Converts events to a DataFrame.
  - Loads any existing files in the target partition (`collection_date` + `source`).
  - Concatenates and drops duplicates by `(symbol, earnings_date, source)`.
  - Rewrites a single consolidated Parquet file for that partition.
  - Other partitions (e.g., previous `collection_date=*`) are never touched.

## Disabled generic save

- To prevent accidental overwrites, `EarningsCalendarRepository.save()` is disabled and will raise.
- The generic `BaseRepository.save()` uses `pyarrow.parquet.write_to_dataset` and is intended for non-earnings datasets. For earnings, always call `store_earnings_batch`.

## Insert-only semantics

- The system is designed to be insert/merge-only within a given partition:
  - New runs add rows for new events.
  - Identical events are coalesced by the dedup key.
  - No global truncation occurs.

## Presence checks

- Before writing, the service performs a read to detect pre-existing events:
  - Reads the repo for the min–max earnings date in the fetched set, filtered by symbols and sources.
  - Skips any event already present by `(symbol, earnings_date, source)`.

## Recovery

- If a partition file is corrupted or needs rebuild:
  - Remove only the target partition folder (e.g., `collection_date=2025-08-23/source=nasdaq`) and re-run the fetch.
  - Other collection dates remain intact.

## Tips

- Keep `collection_date` aligned to the local day of ingestion to make troubleshooting easier.
- Avoid calling low-level Parquet writers directly; always go through the repository.
- Use the `GET /calendar/stats` endpoint to inspect storage size, counts, and date ranges.

