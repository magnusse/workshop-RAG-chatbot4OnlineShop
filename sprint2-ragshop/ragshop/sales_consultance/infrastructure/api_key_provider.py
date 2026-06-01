"""Resolves the WPS API key with a hybrid strategy:

   1. WEBUI_API_KEY_RESET=1 wipes the keyring entry first (force re-prompt).
   2. Keyring (macOS Keychain / Windows Credential Manager / Linux Secret Service).
   3. WEBUI_API_KEY env var (then remembered in keyring for future runs).
   4. Interactive terminal prompt via getpass (then remembered in keyring).
"""

import getpass
import os
import sys

import keyring
import keyring.errors


SERVICE_NAME = "ragshop"
KEY_USERNAME = "wps-api-key"
ENV_VAR = "WEBUI_API_KEY"
RESET_ENV_VAR = "WEBUI_API_KEY_RESET"


def get_wps_api_key() -> str:
    if os.getenv(RESET_ENV_VAR):
        try:
            keyring.delete_password(SERVICE_NAME, KEY_USERNAME)
            print("[api-key] Previous keyring entry deleted (reset requested).")
        except keyring.errors.PasswordDeleteError:
            pass

    stored = keyring.get_password(SERVICE_NAME, KEY_USERNAME)
    if stored:
        return stored

    from_env = os.getenv(ENV_VAR)
    if from_env:
        keyring.set_password(SERVICE_NAME, KEY_USERNAME, from_env)
        print(
            f"[api-key] Picked up from {ENV_VAR}; stored in keyring for future runs."
        )
        return from_env

    entered = _prompt_for_key()
    keyring.set_password(SERVICE_NAME, KEY_USERNAME, entered)
    print("[api-key] Stored in keyring for future runs.")
    return entered


def _prompt_for_key() -> str:
    if not sys.stdin.isatty():
        raise EnvironmentError(
            f"No {ENV_VAR} in environment, keyring is empty, and no TTY is "
            "available to prompt. Run once interactively to seed the keyring, "
            f"or set {ENV_VAR} for this run."
        )
    print(f"[api-key] Not found in keyring or {ENV_VAR}. Please enter the WPS API key.")
    entered = getpass.getpass("WPS API key: ").strip()
    if not entered:
        raise EnvironmentError("Empty API key entered.")
    return entered
