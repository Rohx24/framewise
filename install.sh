#!/usr/bin/env bash
# Installs both skills. They share one codebase, so `replicate` is a thin
# skill directory whose scripts/ points at this repo's.
set -euo pipefail
SKILLS="${1:-$HOME/.claude/skills}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SKILLS"
if [ "$HERE" != "$SKILLS/watch" ]; then
  rm -rf "$SKILLS/watch"
  cp -R "$HERE" "$SKILLS/watch"
fi
mkdir -p "$SKILLS/replicate"
cp "$SKILLS/watch/replicate-skill/SKILL.md" "$SKILLS/replicate/SKILL.md"
ln -sfn ../watch/scripts "$SKILLS/replicate/scripts"

echo "installed:"
echo "  /watch      -> $SKILLS/watch"
echo "  /replicate  -> $SKILLS/replicate"
