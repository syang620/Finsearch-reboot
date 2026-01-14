#!/bin/bash
# Download Apple's 2024 10-K and 2024 Q1 10-Q XBRL files

echo "Downloading Apple 10-K 2024..."
python scripts/download_xbrl.py --ticker AAPL --form 10-K --year 2024 --output-dir data/xbrl

echo ""
echo "Downloading Apple 10-Q 2024 Q1..."
python scripts/download_xbrl.py --ticker AAPL --form 10-Q --year 2024 --quarter 1 --output-dir data/xbrl

echo ""
echo "✅ Done! XBRL files saved to data/xbrl/"
