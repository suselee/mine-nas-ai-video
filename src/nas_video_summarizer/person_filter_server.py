#!/usr/bin/env python3
"""Person filter HTTP service — runs on Armbian.

Start:  person-filter-server [--host 0.0.0.0] [--port 5000]

Endpoints:
    GET  /health          — liveness check
    POST /detect/batch    — bulk person detection
"""
from __future__ import annotations

import argparse
import logging
import os

from flask import Flask, jsonify, request

from .person_filter import PersonFilter

app = Flask(__name__)
_filter: PersonFilter | None = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/detect/batch", methods=["POST"])
def detect_batch():
    data = request.get_json(force=True)
    images = data.get("images", [])
    results: list[dict[str, object]] = []
    for idx, b64 in enumerate(images):
        info = _filter.detect(b64)
        info["idx"] = idx
        results.append(info)
    return jsonify({"scores": results})


def main():
    parser = argparse.ArgumentParser(description="Person filter detection server")
    parser.add_argument("--host", default=os.getenv("PERSON_FILTER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PERSON_FILTER_PORT", "5000")))
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("PERSON_FILTER_THRESHOLD", "0.2")),
    )
    args = parser.parse_args()

    global _filter
    _filter = PersonFilter(threshold=args.threshold)

    logging.basicConfig(level=logging.INFO)
    logging.info("person-filter-server starting on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port)
