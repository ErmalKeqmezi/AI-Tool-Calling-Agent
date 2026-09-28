import pytest

from app.agent import Agent, ToolExecutor
from app.config.settings import Settings
from app.tools import build_registry


@pytest.fixture
def settings(tmp_path):
    return Settings(anthropic_api_key=None, file_sandbox_dir=tmp_path / "workspace")


@pytest.fixture
def registry(settings):
    return build_registry(settings)


@pytest.fixture
def executor(registry):
    return ToolExecutor(registry)


@pytest.fixture
def make_agent(registry):
    def _make(llm, **kwargs):
        return Agent(llm=llm, registry=registry, **kwargs)

    return _make
