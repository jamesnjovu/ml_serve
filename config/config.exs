import Config

# This directory is for developing MLServe itself — it is excluded from the Hex package by the
# `files:` allowlist in mix.exs, so nothing here reaches consumers.
import_config "#{config_env()}.exs"
