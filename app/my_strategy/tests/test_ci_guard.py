"""CI checks application sources while leaving runtime fixtures outside its scope."""
from my_strategy.scripts import ci_guard


def test_syntax_checks_ignore_runtime_fixtures_but_reject_bad_source(tmp_path, monkeypatch):
    app = tmp_path / "app"
    strategy = app / "my_strategy"
    (tmp_path / "install.py").write_text("pass\n", encoding="utf-8")
    for folder in ("data", "artifacts", "services"):
        (strategy / folder).mkdir(parents=True)
    for folder in ("data", "artifacts"):
        (strategy / folder / "fixture.py").write_text("deliberately invalid python\n", encoding="utf-8")
    source = strategy / "services/current.py"
    source.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(ci_guard, "PROJECT_ROOT", app)
    monkeypatch.setattr(ci_guard, "REPO_ROOT", tmp_path)
    assert ci_guard.syntax_checks() == 0
    source.write_text("def broken(:\n", encoding="utf-8")
    assert ci_guard.syntax_checks() == 1
