"""Smoke tests for the embedded-service HTTP API (muser/service.py).

These exercise the JSON/binary endpoints that do NOT need the warm embedding
model, so they stay fast and CI-safe (the model is heavy to load). We build the
app with `create_app()` and drive it with FastAPI's `TestClient` WITHOUT the
`with` context manager — entering the context fires the `@app.on_event("startup")`
handler, which kicks off a background model warm + sidecar priming. Plain
`TestClient(app)` skips startup, so no model is ever loaded here.

Anything that would embed (real `/api/search`, a real image upload) is avoided:
we only assert that the missing-input / validation paths error cleanly (4xx, not
500) before any model work happens.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muser.service import create_app


@pytest.fixture(scope="module")
def client():
    # No `with` → startup event (model warm) never fires. Endpoints that don't
    # touch the embedder work fine against a cold app.
    return TestClient(create_app())


def _jpg(path, color=(180, 40, 40), size=(48, 48)):
    Image.new("RGB", size, color).save(path, "JPEG")
    return str(path)


# --------------------------------------------------------------------------- #
# Route registration                                                          #
# --------------------------------------------------------------------------- #

NEW_ROUTES = [
    "/api/checkout",
    "/api/checkout/status",
    "/api/checkout/zip",
    "/api/pipelines",
    "/api/pipeline/{run_id}",
    "/api/pipeline/{run_id}/file/{path:path}",
    "/api/pipeline/{run_id}/threed",
    "/api/search-upload",
    "/api/search-color",
    "/api/upscale",
    "/api/projection",
    "/api/contact-sheet",
    "/api/demo-mode",
]


def test_routes_registered(client):
    paths = {r.path for r in client.app.routes}
    missing = [r for r in NEW_ROUTES if r not in paths]
    assert not missing, f"missing routes: {missing}"


# --------------------------------------------------------------------------- #
# /api/demo-mode  (no model)                                                   #
# --------------------------------------------------------------------------- #

def test_demo_mode_toggle(client):
    # Capture the starting state so the test restores it (module-global flag).
    start = client.get("/api/demo-mode").json()["hide"]
    try:
        off = client.post("/api/demo-mode", json={"on": False})
        assert off.status_code == 200
        assert off.json() == {"hide": False}
        assert client.get("/api/demo-mode").json() == {"hide": False}

        on = client.post("/api/demo-mode", json={"on": True})
        assert on.status_code == 200
        assert on.json() == {"hide": True}
        assert client.get("/api/demo-mode").json() == {"hide": True}
    finally:
        client.post("/api/demo-mode", json={"on": start})


def test_demo_mode_get_shape(client):
    body = client.get("/api/demo-mode").json()
    assert set(body) == {"hide"}
    assert isinstance(body["hide"], bool)


# --------------------------------------------------------------------------- #
# /api/contact-sheet  (PIL only, no model)                                     #
# --------------------------------------------------------------------------- #

def test_contact_sheet_ok(client, tmp_path):
    a = _jpg(tmp_path / "a.jpg", (200, 30, 30))
    b = _jpg(tmp_path / "b.jpg", (30, 30, 200))
    r = client.post("/api/contact-sheet", json={"paths": [a, b], "cols": 2, "cell": 64})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["X-Sheet-Count"] == "2"
    # Valid JPEG bytes (SOI marker) that PIL can actually open.
    assert r.content[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.format == "JPEG"
        assert im.width > 0 and im.height > 0


def test_contact_sheet_empty_paths_400(client):
    r = client.post("/api/contact-sheet", json={"paths": []})
    assert r.status_code == 400


def test_contact_sheet_all_missing_400(client, tmp_path):
    r = client.post(
        "/api/contact-sheet",
        json={"paths": [str(tmp_path / "nope1.jpg"), str(tmp_path / "nope2.jpg")]},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# /api/pipelines + /api/pipeline/{id}  (sidecar JSON, no model)                #
# --------------------------------------------------------------------------- #

def test_pipelines_list_shape(client):
    r = client.get("/api/pipelines")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)


def test_pipeline_unknown_id_404(client):
    r = client.get("/api/pipeline/definitely-not-a-real-run-id")
    assert r.status_code == 404


def test_pipeline_threed_unknown_id_404(client):
    # Valid body, but the run doesn't exist → 404 (run lookup precedes work).
    r = client.post(
        "/api/pipeline/definitely-not-a-real-run-id/threed",
        json={"output_index": 0},
    )
    assert r.status_code == 404


def test_pipeline_file_path_traversal_guard(client):
    # A `..`-escaping path must never serve a file outside the run dir. The guard
    # returns 403 when the resolved target escapes; a normalized/non-existent
    # path returns 404. Either way: NOT 200, and no /etc/hosts content leaks.
    r = client.get("/api/pipeline/some-run/file/../../../../../../etc/hosts")
    assert r.status_code in (403, 404)
    assert b"localhost" not in r.content  # /etc/hosts content never leaked


# --------------------------------------------------------------------------- #
# Request-model validation (Pydantic ForwardRef regression guard)             #
# --------------------------------------------------------------------------- #
# With `from __future__ import annotations`, a BaseModel defined inside
# create_app() degrades to a query param (FastAPI can't resolve the ForwardRef),
# so a valid JSON body would wrongly 422. These models are module-level, so a
# valid body is accepted and only a *missing required field* yields 422.

def test_checkout_missing_body_422(client):
    # CheckoutReq has no strictly-required field (items defaults to []), but an
    # empty/valid body must reach the handler and 400 on "no valid files" — NOT
    # 422. That proves the body is parsed as JSON, not treated as a query param.
    r = client.post("/api/checkout", json={})
    assert r.status_code == 400  # "no valid files in cart", handler reached


def test_checkout_garbage_body_422(client):
    # Wrong type for a declared field → Pydantic validation 422 (body IS parsed).
    r = client.post("/api/checkout", json={"items": "not-a-list"})
    assert r.status_code == 422


def test_demo_mode_missing_on_422(client):
    # DemoModeReq.on is required → empty body is a 422, and crucially the body is
    # parsed as JSON (a query-param misread would 422 differently / always).
    r = client.post("/api/demo-mode", json={})
    assert r.status_code == 422


def test_threed_missing_output_index_422(client):
    # ThreeDReq.output_index is required; body validation happens before the
    # run lookup, so a missing field is 422 even for a bogus run id.
    r = client.post("/api/pipeline/whatever/threed", json={})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# /api/search-upload  (must error on missing file WITHOUT loading the model)   #
# --------------------------------------------------------------------------- #

def test_search_upload_no_file_422(client):
    # `file: UploadFile = File(...)` is required → FastAPI 422s before any
    # handler code runs, so the embedding model is never warmed.
    r = client.post("/api/search-upload")
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# /api/search-color  (LAB palette, no model) — validation only                 #
# --------------------------------------------------------------------------- #

def test_search_color_bad_hex_400(client):
    # Malformed hex is rejected with 400 before any palette work; no model load.
    r = client.get("/api/search-color", params={"hex": "xyz"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Endpoints that DO need the warm model — explicitly skipped to keep CI fast   #
# --------------------------------------------------------------------------- #

@pytest.mark.skip(reason="real semantic search warms the heavy embedding model; not CI-safe")
def test_search_real_query(client):  # pragma: no cover
    client.get("/api/search", params={"q": "a red square"})
