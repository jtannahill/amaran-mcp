"""Light settings, mesh counters and scenes, kept in ~/.config/amaran-mcp."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, field
import glob
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from typing import Any

CONFIG_DIR = Path(os.environ.get("AMARAN_MCP_HOME", "~/.config/amaran-mcp")).expanduser()
LIGHT_PATH = CONFIG_DIR / "light.json"
STATE_PATH = CONFIG_DIR / "state.json"
SCENES_PATH = CONFIG_DIR / "scenes.json"

DESKTOP_DB_GLOBS = (
    "~/Library/Application Support/amaran Desktop/*_secure_id/amaran.db",
    "~/Library/Application Support/amaran Desktop/*/amaran.db",
)
SIDUS_DIR = Path(__file__).parent / "sidus"

# Our own mesh address. Lights drop any message whose counter is not above the
# last one they saw from that address, so we must never share an address with
# the amaran apps (they send as 0x0001) or with another tool.
SOURCE_ADDRESS_POOL = range(0x7000, 0x7FFF)


class ConfigError(RuntimeError):
    """A problem the user fixes by running or re-running the import."""


@dataclass
class Light:
    name: str
    model: str
    code: str
    node_address: int
    mac: str
    net_key: str
    app_key: str
    source_address: int


@dataclass
class Capabilities:
    hsi: bool = True
    cct: bool = True
    kelvin_min: int = 2700
    kelvin_max: int = 6500
    tint: bool = False
    effects: list[str] = field(default_factory=list)  # Desktop flags, e.g. "cop_car"


def _write_private(path: Path, data: Any) -> None:
    """Write JSON readable only by this user; the light file holds mesh keys."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def load_light() -> Light:
    data = _read(LIGHT_PATH, None)
    if data is None:
        raise ConfigError(
            f"No light configured ({LIGHT_PATH} is missing). "
            "Pair the light in amaran Desktop, then run: amaran-mcp import"
        )
    return Light(**data)


def capabilities(light: Light) -> Capabilities:
    products = json.loads((SIDUS_DIR / "product_capabilities.json").read_text())["products"]
    flags = products.get(light.code.upper())
    if flags is None:
        return Capabilities()
    return Capabilities(
        hsi=str(flags.get("hsi_support")) == "1",
        cct=str(flags.get("cct_support")) == "1",
        # The catalog stores colour temperature in hundreds of kelvin.
        kelvin_min=int(flags.get("product_cct_min") or 27) * 100,
        kelvin_max=int(flags.get("product_cct_max") or 65) * 100,
        tint=str(flags.get("gm_support")) == "1",
        effects=list(flags.get("system_effects") or []),
    )


class StateStore:
    """Mesh counters that must survive restarts, plus the remembered device."""

    def __init__(self) -> None:
        data = _read(STATE_PATH, {})
        self.next_seq: int = data.get("next_seq", 1)
        self.iv_index: int = data.get("iv_index", 0)
        self.ble_address: str | None = data.get("ble_address")

    def save(self) -> None:
        _write_private(STATE_PATH, {
            "next_seq": self.next_seq,
            "iv_index": self.iv_index,
            "ble_address": self.ble_address,
        })

    def reserve(self, count: int) -> range:
        """Hand out sequence numbers, saving before any of them are used."""

        first = self.next_seq
        if first + count > 0xFFFFFF:
            raise ConfigError("Mesh sequence numbers are used up; re-run amaran-mcp import")
        self.next_seq = first + count
        self.save()
        return range(first, first + count)

    def set_iv_index(self, iv_index: int) -> None:
        if iv_index != self.iv_index:
            self.iv_index = iv_index
            self.save()

    def remember_device(self, address: str) -> None:
        if address != self.ble_address:
            self.ble_address = address
            self.save()


def load_scenes() -> dict[str, dict[str, Any]]:
    return _read(SCENES_PATH, {})


def save_scenes(scenes: dict[str, dict[str, Any]]) -> None:
    _write_private(SCENES_PATH, scenes)


# ── Import from amaran Desktop ───────────────────────────────────────────────
# Adapted from amaran-hacs scripts/export_amaran.py: same tables and columns,
# trimmed to the one-light case.


def find_desktop_db() -> Path:
    for pattern in DESKTOP_DB_GLOBS:
        hits = [Path(p) for p in glob.glob(str(Path(pattern).expanduser()))]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    raise ConfigError(
        "amaran Desktop database not found. Install amaran Desktop, pair the light "
        "in it, then run this again (or pass --db PATH)."
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})")}


def _normalize_key(value: Any) -> str:
    key = value.hex() if isinstance(value, bytes) else str(value or "")
    key = re.sub(r"[^0-9a-fA-F]", "", key.removeprefix("0x")).lower()
    if len(key) != 32:
        raise ConfigError("The amaran Desktop database has a malformed mesh key")
    return key


def _normalize_mac(value: Any) -> str:
    compact = re.sub(r"[^0-9A-Fa-f]", "", str(value or ""))
    if len(compact) != 12:
        return str(value or "")
    return ":".join(compact[i : i + 2] for i in range(0, 12, 2)).upper()


def _catalog_product(product_id: Any, code: str) -> dict[str, Any] | None:
    products = json.loads((SIDUS_DIR / "product.json").read_text())
    for product in products:
        if code and str(product.get("hex", "")).upper() == code.upper():
            return product
    for product in products:
        if product_id is not None and str(product.get("id")) == str(product_id):
            return product
    return None


def list_desktop_fixtures(db_path: Path) -> list[dict[str, Any]]:
    """Return every light in the Desktop database with its network's keys."""

    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        mesh_table = "mesh" if _columns(conn, "mesh") else "meshs"
        mesh_cols = _columns(conn, mesh_table)
        order = " order by update_time desc" if "update_time" in mesh_cols else ""
        meshes = [
            m for m in conn.execute(f"select * from {mesh_table}{order}").fetchall()
            if m["net_key"] and m["app_key"]
        ]
        if not meshes:
            raise ConfigError("amaran Desktop has no mesh network yet; pair the light first")
        fixture_cols = _columns(conn, "fixtures")
        rows = conn.execute("select * from fixtures order by node_address").fetchall()

    fixtures = []
    for row in rows:
        node_address = row["node_address"]
        if node_address is None or int(node_address) < 2 or not row["mac_address"]:
            continue
        mesh = meshes[0]
        if "mesh_uuid" in fixture_cols and row["mesh_uuid"]:
            mesh = next((m for m in meshes if m["uuid"] == row["mesh_uuid"]), mesh)
        code = str(row["code"] or "").strip() if "code" in fixture_cols else ""
        product_id = row["product_id"] if "product_id" in fixture_cols else None
        product = _catalog_product(product_id, code)
        fixtures.append({
            "name": " ".join(str(row["name"] or "").split()) if "name" in fixture_cols else "",
            "model": (product or {}).get("name", "Unknown"),
            "code": code or (product or {}).get("hex", ""),
            "node_address": int(node_address),
            "mac": _normalize_mac(row["mac_address"]),
            "net_key": _normalize_key(mesh["net_key"]),
            "app_key": _normalize_key(mesh["app_key"]),
        })
    if not fixtures:
        raise ConfigError("amaran Desktop has no paired lights yet")
    return fixtures


# ── Import from the amaran account ───────────────────────────────────────────
# The amaran apps sync every light and its mesh keys to the Sidus account
# service. Endpoints, headers and region rules are from amaran-hacs cloud.py.

SIDUS_HEADERS = {"platform": "2", "request_source": "4", "Content-Type": "application/json"}
EU_COUNTRIES = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
}
CN_COUNTRIES = {"CN", "HK", "MO", "TW"}


def _sidus_request(url: str, *, body: dict[str, Any] | None = None, token: str | None = None) -> dict[str, Any]:
    from urllib.request import Request, urlopen

    headers = {**SIDUS_HEADERS, **({"Token": token} if token else {})}
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(url, data=data, headers=headers), timeout=20) as response:  # noqa: S310
        reply = json.loads(response.read())
    if not isinstance(reply, dict):
        raise ConfigError("The amaran account service sent an unexpected reply")
    return reply


def list_cloud_fixtures(account: str, password: str, country: str = "US") -> list[dict[str, Any]]:
    """Sign in to the amaran account and return its lights with their keys."""

    area = country.upper()
    for _ in range(2):
        region = "cn" if area in CN_COUNTRIES else "eu" if area in EU_COUNTRIES else "us"
        base = f"https://{region}.sidus.link/"
        reply = _sidus_request(base + "uc/user/login", body={
            "email": account if "@" in account else "",
            "phone_number": "" if "@" in account else account,
            "password": password,
            "area": area,
        })
        data = reply.get("data") or {}
        if reply.get("code") == 20000 and data.get("token"):
            token = str(data["token"])
            break
        if reply.get("code") in (20002, 20014):
            raise ConfigError("The amaran account rejected that email/phone and password")
        # 4001: the account lives in another region; the reply says which.
        others = [str(u["area"]).upper() for u in data.get("OtherRegionUser") or [] if u.get("area")]
        if reply.get("code") != 4001 or not others:
            raise ConfigError(f"amaran sign-in failed (code {reply.get('code')})")
        area = others[0]
    else:
        raise ConfigError("amaran sign-in kept moving between regions")

    reply = _sidus_request(base + "amaran/pack/query", token=token)
    data = reply.get("data")
    if reply.get("code") != 20000 or not isinstance(data, dict):
        raise ConfigError(f"Could not read lights from the amaran account (code {reply.get('code')})")
    meshes = {
        m.get("mesh_uuid"): m
        for m in (data.get("meshs") or {}).get("existing") or []
        if m.get("net_key") and m.get("app_key")
    }
    fixtures = []
    for row in (data.get("fixtures") or {}).get("existing") or []:
        mesh = meshes.get(row.get("mesh_uuid"))
        node_address = row.get("node_address")
        if mesh is None or node_address is None or int(node_address) < 2:
            continue
        code = str(row.get("code") or "")
        product = _catalog_product(row.get("product_id"), code)
        fixtures.append({
            "name": " ".join(str(row.get("name") or "").split()),
            "model": (product or {}).get("name", "Unknown"),
            "code": code or (product or {}).get("hex", ""),
            "node_address": int(node_address),
            "mac": _normalize_mac(row.get("mac_address")),
            "net_key": _normalize_key(mesh["net_key"]),
            "app_key": _normalize_key(mesh["app_key"]),
        })
    if not fixtures:
        raise ConfigError(
            "The amaran account has no lights. If you paired with Sidus Link rather "
            "than the amaran app, pair the light in amaran Desktop instead."
        )
    return fixtures


def import_light(fixture: dict[str, Any]) -> Light:
    """Save a light from list_desktop_fixtures, keeping our address if unchanged."""

    previous = _read(LIGHT_PATH, None)
    if previous and previous["net_key"] == fixture["net_key"]:
        source_address = previous["source_address"]
    else:
        source_address = random.choice(SOURCE_ADDRESS_POOL)
        # A new network or address starts a fresh counter.
        _write_private(STATE_PATH, {"next_seq": 1, "iv_index": 0, "ble_address": None})
    light = Light(source_address=source_address, **fixture)
    if not light.name:
        light.name = light.model
    _write_private(LIGHT_PATH, asdict(light))
    return light
