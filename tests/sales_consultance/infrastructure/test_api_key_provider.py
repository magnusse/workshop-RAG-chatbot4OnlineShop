import pytest

from ragshop.sales_consultance.infrastructure import api_key_provider


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.delenv(api_key_provider.ENV_VAR, raising=False)
    monkeypatch.delenv(api_key_provider.RESET_ENV_VAR, raising=False)


def _patch_keyring(monkeypatch, *, stored=None):
    """Patch the three keyring entrypoints used by the provider with a tiny
    in-memory backend. Returns the backing dict so tests can introspect writes."""
    store: dict = {}
    if stored is not None:
        store[(api_key_provider.SERVICE_NAME, api_key_provider.KEY_USERNAME)] = stored

    def get_password(service, username):
        return store.get((service, username))

    def set_password(service, username, password):
        store[(service, username)] = password

    def delete_password(service, username):
        if (service, username) not in store:
            import keyring.errors

            raise keyring.errors.PasswordDeleteError("not found")
        del store[(service, username)]

    monkeypatch.setattr(api_key_provider.keyring, "get_password", get_password)
    monkeypatch.setattr(api_key_provider.keyring, "set_password", set_password)
    monkeypatch.setattr(api_key_provider.keyring, "delete_password", delete_password)
    return store


def test_returns_value_from_keyring_when_present(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch, stored="stored-key")

    assert api_key_provider.get_wps_api_key() == "stored-key"


def test_falls_back_to_env_var_and_caches_in_keyring(isolated_env, monkeypatch):
    store = _patch_keyring(monkeypatch)
    monkeypatch.setenv(api_key_provider.ENV_VAR, "env-key")

    result = api_key_provider.get_wps_api_key()

    assert result == "env-key"
    assert store[
        (api_key_provider.SERVICE_NAME, api_key_provider.KEY_USERNAME)
    ] == "env-key"


def test_prompts_when_keyring_and_env_empty(isolated_env, monkeypatch):
    store = _patch_keyring(monkeypatch)
    monkeypatch.setattr(api_key_provider, "_prompt_for_key", lambda: "typed-key")

    result = api_key_provider.get_wps_api_key()

    assert result == "typed-key"
    assert store[
        (api_key_provider.SERVICE_NAME, api_key_provider.KEY_USERNAME)
    ] == "typed-key"


def test_reset_flag_deletes_keyring_entry_and_re_prompts(isolated_env, monkeypatch):
    store = _patch_keyring(monkeypatch, stored="old-key")
    monkeypatch.setenv(api_key_provider.RESET_ENV_VAR, "1")
    monkeypatch.setattr(api_key_provider, "_prompt_for_key", lambda: "new-key")

    result = api_key_provider.get_wps_api_key()

    assert result == "new-key"
    assert store[
        (api_key_provider.SERVICE_NAME, api_key_provider.KEY_USERNAME)
    ] == "new-key"


def test_reset_flag_is_idempotent_when_keyring_empty(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch)
    monkeypatch.setenv(api_key_provider.RESET_ENV_VAR, "1")
    monkeypatch.setattr(api_key_provider, "_prompt_for_key", lambda: "fresh-key")

    assert api_key_provider.get_wps_api_key() == "fresh-key"


def test_prompt_raises_without_tty(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch)
    monkeypatch.setattr(api_key_provider.sys.stdin, "isatty", lambda: False)

    with pytest.raises(EnvironmentError, match="TTY"):
        api_key_provider.get_wps_api_key()


def test_prompt_rejects_empty_input(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch)
    monkeypatch.setattr(api_key_provider.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        api_key_provider.getpass, "getpass", lambda prompt="": "   "
    )

    with pytest.raises(EnvironmentError, match="Empty"):
        api_key_provider.get_wps_api_key()
