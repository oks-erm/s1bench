#!/bin/zsh
cd "$(dirname "$0")" || exit 1
print "Choose a model (run only one at a time on this Mac):"
print "1 — Laya"
print "2 — Nimble"
print "3 — CLM 8-bit MLX community port"
read "selection?Enter 1, 2, or 3: "
case "$selection" in
  1) model=laya ;;
  2) model=nimble ;;
  3) model=clm ;;
  *) print "No model selected."; exit 1 ;;
esac
.venv/bin/python local_models.py "$model"
read "finished?Press Return to close. "
