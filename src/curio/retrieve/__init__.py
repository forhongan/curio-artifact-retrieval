"""Retrieval modules; lightweight API imports do not load the search engine."""

__all__ = ["SearchService"]


def __getattr__(name):
    if name == "SearchService":
        from .engine import SearchService

        return SearchService
    raise AttributeError(name)
