"""The eval GUI needs, and offers the Router, only the keys of the LLMs its Policy Server serves."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def gui(tmp_path, monkeypatch):
    if sys.version_info < (3, 12):
        pytest.skip('Existing review_evals.py uses Python 3.12 f-string syntax')
    pytest.importorskip('fastapi', reason='The GUI is an optional extra')
    pytest.importorskip('uvicorn', reason='The GUI is an optional extra')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('eval_gui')
    monkeypatch.setattr(module, 'ARGS', SimpleNamespace(server_dir=tmp_path, server_config='server.yaml'),
                        raising=False)  # main() sets it
    monkeypatch.setattr(module, 'S', module.State())
    monkeypatch.setattr(module, 'KEYS', {})
    monkeypatch.setattr(module, 'KEY_OK', set())
    monkeypatch.setattr(module, 'verify_key', lambda provider, key: (True, 'works'))
    return module


def serve(gui, *models):
    lines = ['models:'] + [f'  - {{name: {name}, model_type: {kind}}}' for name, kind in models]
    (gui.ARGS.server_dir / gui.ARGS.server_config).write_text('\n'.join(lines) + '\n')


def test_served_providers_follow_model_names(gui):
    serve(gui, ('g05-yam', 'vla'), ('grok-4.7', 'llm'), ('claude-opus-5-5-capx', 'llm'),
          ('gemini-4-hydrogen', 'llm'))
    assert gui.llm_providers_served() == {'xai', 'anthropic', 'google'}


def test_missing_gemini_key_blocks_only_when_gemini_is_served(gui):
    gui.KEYS.update({'xai': 'grok-key', 'anthropic': 'claude-key'})
    serve(gui, ('grok-4.7', 'llm'), ('claude-opus-5-5', 'llm'))
    gui.check_keys()
    assert gui.S.checks['keys']['status'] == 'ok'
    assert [row['provider'] for row in gui.S.keys] == ['xai', 'anthropic']

    serve(gui, ('grok-4.7', 'llm'), ('claude-opus-5-5', 'llm'), ('gemini-4-hydrogen', 'llm'))
    gui.check_keys()
    assert gui.S.checks['keys']['status'] == 'fail'
    assert 'Google · Gemini key missing' in gui.S.checks['keys']['detail']


def test_gemini_key_is_saved_and_checked(gui, tmp_path, monkeypatch):
    gui.ARGS.llm_keys = tmp_path / 'llm-keys.env'
    monkeypatch.setenv('GEMINI_API_KEY', '')  # save_keys exports it; restored after the test
    gui.save_keys({'google': ' gemini-key '})
    assert gui.ARGS.llm_keys.read_text() == 'export GEMINI_API_KEY=gemini-key\n'
    serve(gui, ('gemini-4-hydrogen', 'llm'))
    gui.KEYS['anthropic'] = 'claude-key'
    gui.check_keys()
    assert gui.S.checks['keys']['status'] == 'ok'
    assert gui.KEY_OK == {'google', 'anthropic'}
