EVAL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "global": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string"},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                "seed": {"type": "integer"},
            },
            "required": ["output_dir", "log_level"],
        },
        "retrieval": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "minimum": 1},
                "score_threshold": {"type": "number", "minimum": 0.0},
                "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "search_type": {
                    "type": "string",
                    "enum": ["weighted", "rrf", "two_stage"],
                },
            },
            "required": ["k", "score_threshold"],
        },
        "generation": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "minimum": 1},
                "score_threshold": {"type": "number", "minimum": 0.0},
                "limit": {"type": ["integer", "null"]},
            },
            "required": ["k", "score_threshold"],
        },
        "ab_testing": {
            "type": "object",
            "properties": {
                "configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "chunk_size": {"type": "integer", "minimum": 1},
                            "chunk_overlap": {"type": "integer", "minimum": 0},
                            "embedding_model": {"type": "string"},
                            "k": {"type": "integer", "minimum": 1},
                            "score_threshold": {"type": "number", "minimum": 0.0},
                        },
                        "required": ["name", "chunk_size", "chunk_overlap", "k"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["configs"],
        },
    },
    "required": ["global"],
}
