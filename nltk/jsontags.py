# Natural Language Toolkit: JSON Encoder/Decoder Helpers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Xu <xxu@student.unimelb.edu.au>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Register JSON tags, so the nltk data loader knows what module and class to look for.

NLTK uses simple '!' tags to mark the types of objects, but the fully-qualified
"tag:nltk.org,2011:" prefix is also accepted in case anyone ends up
using it.
"""

import json
from functools import lru_cache
from typing import Any

json_tags = {}

TAG_PREFIX = "!"


def register_tag(cls):
    """
    Decorates a class to register it's json tag.
    """
    json_tags[TAG_PREFIX + getattr(cls, "json_tag")] = cls
    return cls


class JSONTaggedEncoder(json.JSONEncoder):
    def default(self, obj):
        obj_tag = getattr(obj, "json_tag", None)
        if obj_tag is None:
            return super().default(obj)
        obj_tag = TAG_PREFIX + obj_tag
        obj = obj.encode_json_obj()
        return {obj_tag: obj}


class JSONTaggedDecoder(json.JSONDecoder):
    def decode(self, s: str) -> Any:
        return self.decode_obj(super().decode(s))

    @classmethod
    def decode_obj(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            obj = {key: cls.decode_obj(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            obj = [cls.decode_obj(val) for val in obj]

        if not isinstance(obj, dict) or len(obj) != 1:
            return obj

        obj_tag = next(iter(obj.keys()))
        if not obj_tag.startswith("!"):
            return obj

        obj_cls = cls._cached_decode_tag(obj_tag)
        return obj_cls.decode_json_obj(obj[obj_tag])

    @lru_cache(maxsize=256)
    def _cached_decode_tag(tag: str) -> Any:
        if tag not in json_tags:
            raise ValueError("Unknown tag", tag)
        return json_tags[tag]


__all__ = ["register_tag", "json_tags", "JSONTaggedEncoder", "JSONTaggedDecoder"]
