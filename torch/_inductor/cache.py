from __future__ import annotations

from abc import ABC, abstractmethod
from ast import literal_eval
from functools import cached_property
from hashlib import sha256
from os import getenv
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, Self, override

from torch.utils._filelock import FileLock

import pickle

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


class CacheError(ValueError):
    pass


Key = TypeVar("Key", str, int, tuple["Key", ...])
Value = TypeVar("Value", str, int, tuple["Value", ...], bytes, dict["Value", "Value"], list["Value"])


class Cache(ABC, Generic[Key, Value]):
    @abstractmethod
    def get(self: Self, key: Key) -> Value | None:
        pass

    @abstractmethod
    def insert(self: Self, key: Key, value: Value) -> bool:
        pass


class InMemoryCache(Cache[Key, Value]):
    def __init__(self: Self) -> None:
        self._cache: dict[Key, Value] = {}
        self._lock: Lock = Lock()

    def get(self: Self, key: Key) -> Value | None:
        with self._lock:
            if (value := self._cache.get(key)) is not None:
                return value
            return None

    def insert(self: Self, key: Key, value: Value) -> bool:
        with self._lock:
            if key in self._cache:
                # no overwrites for insert!
                return False
            self._cache[key] = value
            return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        cache = cls()

        if (env_val := getenv(env_var)) is None:
            # env_var doesn't exist = empty cache
            return cache

        for kv_pair in env_val.split(";"):
            # ignore whitespace prefix/suffix
            kv_pair = kv_pair.strip()

            if not kv_pair:
                # kv_pair could be '' if env_val is '' or has ; suffix
                continue

            try:
                key_bytes_repr, value_bytes_repr = kv_pair.split(",", 1)
                # ignore whitespace prefix/suffix, again
                key_bytes_repr, value_bytes_repr = key_bytes_repr.strip(), value_bytes_repr.strip()
            except ValueError as err:
                raise CacheError(
                    f"Malformed kv_pair {kv_pair!r} from env_var {env_var!r}, likely missing comma separator."
                ) from err

            try:
                # check that key_bytes_str is an actual, legitimate encoding
                key_bytes = literal_eval(key_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err
            try:
                # check that value_bytes_str is an actual, legitimate encoding
                value_bytes = literal_eval(value_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err

            try:
                key = pickle.loads(key_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err
            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err

            # true duplicates, i.e. multiple occurences of the same key => value
            # mapping are ok and treated as a no-op; key duplicates with differing
            # values, i.e. key => value_1 and key => value_2 where value_1 != value_2,
            # are not okay since we don't allow overwriting cached values (it's bad regardless)
            if (not cache.insert(key, value)) and (cache.get(key) != value):
                raise CacheError(
                    f"Multiple values for key {key!r} found, got {cache.get(key)!r} and {value!r}."
                )

        return cache

    @classmethod
    def from_file_path(cls, fpath: Path) -> Self:
        cache = cls()

        if not fpath.is_file():
            # fpath doesn't exit = empty cache
            return cache

        try:
            with open(fpath, "rb") as fp:
                cache._cache = pickle.load(fp)
                if not isinstance(cache._cache, dict):
                    raise CacheError(
                        f"Failed to create cache from file path {fpath}, file contents not pickled dict[Key, Value]."
                    )
                assert isinstance(cache._cache, dict)
        except pickle.UnpicklingError as err:
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents are un-pickle-able."
            ) from err

        return cache


class AsyncCache(Cache[Key, Value]):
    def get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        return executor.submit(self.get, key)

    def insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        return executor.submit(self.insert, key, value)


class OnDiskCache(AsyncCache[Key, Value]):
    @cached_property
    def base_dir(self: Self) -> Path:
        return Path(gettempdir()) / "cache"

    def _fpath_from_key(self: Self, key: Key) -> Path:
        try:
            return self.base_dir / sha256(pickle.dumps(key)).hexdigest()[:32]
        except (AttributeError, pickle.PicklingError) as err:
            raise CacheError(
                f"Failed to get fpath for key {key!r}, key is not pickle-able."
            ) from err
        assert_never(key)

    def _flock_from_fpath(self: Self, fpath: Path) -> FileLock:
        # fpath.name is a hex digest, meaning there are 16^4 potential values
        # for fpath.name[:4]; this is more than enough unique locks to not
        # cause additional overhead from shared locks and it also saves our
        # cache dir from becoming 50 percent locks
        return FileLock(str(fpath.parent / "locks" / fpath.name[:4]) + ".lock")

    @override
    def get(self: Self, key: Key) -> Value | None:
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)

        with flock:
            if not fpath.is_file():
                return None

            try:
                with open(fpath, "rb") as fp:
                    return pickle.load(fp)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Failed to get key {key!r}, value is potentially corrupted (value is not un-pickle-able)."
                ) from err

    @override
    def insert(self: Self, key: Key, value: Value) -> bool:
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite); use
            # flock for added atomicity guarantee and to prevent partial writes
            with flock as _, open(fpath, "xb") as fp:
                pickle.dump(value, fp)
        except pickle.PicklingError as err:
            raise CacheError(
                f"Failed to insert key {key!r} with value {value!r}, value is not pickle-able."
            ) from err
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache[Key, Value]):
    @cached_property
    def base_dir(self: Self) -> Path:
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache")
