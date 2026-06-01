import pytest

from ragshop.sales_consultance.infrastructure import api_key_provider


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.delenv(api_key_provider.ENV_VAR, raising=False)
    monkeypatch.delenv(api_key_provider.RESET_ENV_VAR, raising=False)


def _patch_keyring(monkeypatch, *, stored=None):
    store: dict = {}
    if stored is not None:
        store[(api_key_provider.SERVICE_NAME, api_key_provider.KEY_USERNAME)] = stored
    monkeypatch.setattr(
        api_key_provider.keyring, "get_password",
        lambda s, u: store.get((s, u)),
    )
    monkeypatch.setattr(
        api_key_provider.keyring, "set_password",
        lambda s, u, p: store.__setitem__((s, u), p),
    )

    def delete(s, u):
        if (s, u) not in store:
            import keyring.errors

            raise keyring.errors.PasswordDeleteError("not found")
        del store[(s, u)]

    monkeypatch.setattr(api_key_provider.keyring, "delete_password", delete)
    return store


def test_returns_value_from_keyring(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch, stored="stored-key")
    assert api_key_provider.get_wps_api_key() == "stored-key"


def test_env_var_fallback(isolated_env, monkeypatch):
    _patch_keyring(monkeypatch)
    monkeypatch.setenv(api_key_provider.ENV_VAR, "env-key")
    assert api_key_provider.get_wps_api_key() == "env-key"
