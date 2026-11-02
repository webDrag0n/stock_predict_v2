# Contributing / 贡献指南

感谢对本项目的关注！本文件同时提供英文与中文说明。

Thanks for your interest in contributing! This file is bilingual (English / 中文).

---

## English

### Development setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for environment setup (`uv sync`), running
tests, and project conventions.

### Before submitting a PR

1. **Tests pass**: run `uv run pytest` — all tests must pass.
2. **Code style**: the project uses `black` and `ruff` with `line-length = 100`
   (configured in `pyproject.toml`). Run `uv run black .` and
   `uv run ruff check src pipeline tests scripts` before committing.
3. **No secrets**: never commit `config_credentials.toml`, `data_cache/`,
   `outputs/`, or `models/` — they are gitignored for a reason.
4. **Keep changes focused**: one concern per PR; describe the motivation and
   the verification you ran.

### Reporting issues

Please include: OS, Python version (`python --version`), the exact command you
ran, the full traceback, and whether the data cache was warm or cold.

---

## 中文

### 开发环境

环境搭建（`uv sync`）、测试运行与项目约定见 [DEVELOPMENT.md](DEVELOPMENT.md)。

### 提交 PR 前

1. **测试通过**：运行 `uv run pytest`，全部测试必须通过。
2. **代码风格**：项目使用 `black` 与 `ruff`，`line-length = 100`（配置见
   `pyproject.toml`）。提交前运行 `uv run black .` 和
   `uv run ruff check src pipeline tests scripts`。
3. **勿提交敏感与产物文件**：`config_credentials.toml`、`data_cache/`、
   `outputs/`、`models/` 已被 gitignore，请勿强制添加。
4. **改动聚焦**：一个 PR 只解决一个问题，并说明动机与你做过的验证。

### 报告问题

请附上：操作系统、Python 版本（`python --version`）、执行的完整命令、
完整报错堆栈，以及数据缓存是冷启动还是已有缓存。
