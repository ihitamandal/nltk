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
    def decode(self, s):
        return self.decode_obj(super().decode(s))

    @classmethod
    def decode_obj(cls, obj):
        if isinstance(obj, dict):
            result = {}
            for key, val in obj.items():
                decoded_val = cls.decode_obj(val)
                # If there is only one item and it is a tag
                if key.startswith("!") and len(obj) == 1:
                    if key not in json_tags:
                        raise ValueError("Unknown tag", key)
                    return json_tags[key].decode_json_obj(decoded_val)
                result[key] = decoded_val
            return result
        elif isinstance(obj, list):
            return [cls.decode_obj(val) for val in obj]
        return obj


__all__ = ["register_tag", "json_tags", "JSONTaggedEncoder", "JSONTaggedDecoder"]
