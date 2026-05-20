#!/usr/bin/env bash
# migrate_to_ai_trading_bot_v2.sh
# Run from ANYWHERE on your Mac. Copies trading-bot code into AI_Trading_Bot_V2.
set -e

SOURCE_REPO="https://github.com/andorszamek/autos-jatek-gitignore"
SOURCE_BRANCH="claude/ai-trading-bot-v2-oxSBH"
TARGET_REPO="https://github.com/andorszamek/AI_Trading_Bot_V2"
TMP_SRC="/tmp/trading_src_$$"
TMP_DST="/tmp/trading_dst_$$"

echo "==> Fetching source code from autos-jatek-gitignore..."
git clone --depth 1 -b "$SOURCE_BRANCH" "$SOURCE_REPO" "$TMP_SRC"

echo "==> Cloning AI_Trading_Bot_V2..."
git clone "$TARGET_REPO" "$TMP_DST"

echo "==> Copying trading-bot contents..."
# Copy everything including hidden files (except .git)
rsync -av --exclude='.git' "$TMP_SRC/trading-bot/" "$TMP_DST/"

echo "==> Committing and pushing..."
cd "$TMP_DST"
git add -A
if git diff --cached --quiet; then
    echo "Nothing to commit — repo already up to date."
else
    git commit -m "feat: complete AI trading bot Phases 1-8 (35 tests green)"
    git push origin main
    echo ""
    echo "==> Done! Check: https://github.com/andorszamek/AI_Trading_Bot_V2"
fi

rm -rf "$TMP_SRC" "$TMP_DST"
