#!/usr/bin/env bash
# Installer for whisper-subtitles:
#   - symlinks bin/whisper-subtitles into ~/.local/bin
#   - seeds ~/.config/whisper-subtitles.env from .env.example (chmod 600)
#   - adds a zsh `noglob` alias so unquoted paths with spaces/[brackets] work
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BIN_DIR="$HOME/.local/bin"
ENV_FILE="$HOME/.config/whisper-subtitles.env"

mkdir -p "$BIN_DIR" "$HOME/.config"

# 1) command on PATH
ln -sf "$REPO/bin/whisper-subtitles" "$BIN_DIR/whisper-subtitles"
chmod +x "$REPO/bin/whisper-subtitles"
echo "Linked: $BIN_DIR/whisper-subtitles -> $REPO/bin/whisper-subtitles"

# 2) key file (don't overwrite an existing one)
if [ ! -f "$ENV_FILE" ]; then
    install -m 600 "$REPO/.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE (chmod 600) — edit it and add your OPENAI_API_KEY."
else
    echo "Kept existing $ENV_FILE"
fi

# 3) zsh alias for unquoted paths (optional; only if ~/.zshrc exists)
if [ -f "$HOME/.zshrc" ] && ! grep -q "alias whisper-subtitles=" "$HOME/.zshrc"; then
    printf '\n# whisper-subtitles: allow unquoted paths with spaces/[brackets]\nalias whisper-subtitles='\''noglob whisper-subtitles'\''\n' >> "$HOME/.zshrc"
    echo "Added zsh alias to ~/.zshrc (open a new terminal to use it)."
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "NOTE: $BIN_DIR is not on your PATH — add it in your shell rc." ;;
esac

echo "Done. Edit $ENV_FILE, open a new terminal, then: whisper-subtitles <video>"
