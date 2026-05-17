#!/usr/bin/env bash
# Run this ONCE on your Mac to populate the AI_Trading_Bot_V2 repo.
# Usage: bash migrate_to_ai_trading_bot_v2.sh
set -e

REPO="https://github.com/andorszamek/AI_Trading_Bot_V2.git"
TMP=$(mktemp -d)

echo "Cloning AI_Trading_Bot_V2..."
git clone "$REPO" "$TMP"

echo "Copying trading-bot contents..."
cp -r ./* "$TMP"/
cp .gitignore "$TMP"/.gitignore
cp .env.example "$TMP"/.env.example

cd "$TMP"
git add .
git commit -m "Initial commit: complete AI trading bot (Phases 1-8)"
git push origin main

echo ""
echo "Done! Code is now in andorszamek/AI_Trading_Bot_V2"
rm -rf "$TMP"
