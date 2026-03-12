#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Stopping and removing containers, networks, and volumes..."
docker compose down --volumes --remove-orphans

read -rp "Also remove the built Docker images? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
  echo "==> Removing images..."
  docker compose down --rmi all --volumes --remove-orphans
  echo "    Images removed."
fi

echo "    Done. Run ./start.sh to rebuild from source."
