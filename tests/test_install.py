"""How the listener's login item runs, for each way of installing."""

from pathlib import Path

from amaran_mcp import SOURCE, _listener_command


def code_in(root: Path) -> Path:
    return root / "src" / "amaran_mcp" / "__init__.py"


def found(**paths):
    return lambda name: paths.get(name)


def test_a_clone_runs_its_own_code(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    argv = _listener_command(code_in(tmp_path), found(uv="/bin/uv", **{"amaran-mcp": "/bin/amaran-mcp"}))
    assert argv == ["/bin/uv", "--directory", str(tmp_path), "run", "amaran-mcp", "listen"]


def test_a_tool_install_runs_its_command(tmp_path):
    site = tmp_path / "lib" / "python3.13" / "site-packages"
    argv = _listener_command(site / "amaran_mcp" / "__init__.py", found(**{"amaran-mcp": "/home/.local/bin/amaran-mcp"}))
    assert argv == ["/home/.local/bin/amaran-mcp", "listen"]


def test_a_cached_copy_runs_the_published_repo(tmp_path):
    argv = _listener_command(code_in(tmp_path), found(uvx="/bin/uvx"))
    assert argv == ["/bin/uvx", "--from", SOURCE, "amaran-mcp", "listen"]
