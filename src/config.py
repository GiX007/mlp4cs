"""Main configuration for MLP4CS."""
from pathlib import Path


# Dataset Paths
MULTIWOZ_DIR = Path("data/multiwoz_github")
MULTIWOZ_DATA_DIR = MULTIWOZ_DIR / "data/MultiWOZ_2.2"
DB_DIR = MULTIWOZ_DIR / "db"
DIALOG_ACTS_FILE = MULTIWOZ_DATA_DIR / "dialog_acts.json"

# Results Paths
RESULTS_DIR: Path = Path("results")
LOGS_DIR: Path = RESULTS_DIR / "logs"


# Downloaded HuggingFace model weights
MODELS_DIR: Path = Path("data/models")


# Domains (full set: hotel, restaurant, taxi, train, attraction, bus, hospital, police)
TARGET_DOMAINS: set[str] = {"hotel", "restaurant"}


# Fine-tuning data and models paths
FINETUNE_DIR = Path("data/finetune_data")
FINETUNE_DST_FILE = FINETUNE_DIR / "dst_train.json"
FINETUNE_RESPGEN_FILE = FINETUNE_DIR / "respgen_train.json"
FINETUNED_MODELS_DIR = Path("data/finetuned_models")


# LLM call defaults
LLM_MAX_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.0  # deterministic outputs for task-oriented pipeline

# Local model loading settings
# LOCAL_MAX_SEQ_LENGTH: int = 8192
LOCAL_MAX_SEQ_LENGTH: int = 32768  # EuroHPC A100 40GB
# LOCAL_MAX_SEQ_LENGTH: int = 65536
LOCAL_LOAD_IN_4BIT: bool = True  # matches bnb-4bit downloaded models
LOCAL_DTYPE = None  # auto-detect: float16 on A100, T4

# Fine-tuning instructions
DST_INSTRUCTION = "You are a dialogue state tracker for a task-oriented dialogue system. Extract ONLY slots explicitly mentioned by the user. Always output in the exact format requested. Never add explanations."
RESPGEN_INSTRUCTION = "Generate a response for the customer service system based on the conversation context."

# Fine-tuning epochs
FINETUNE_EPOCHS: int = 3

# Set to an integer to limit dialogues processed, None = full split
MAX_DIALOGUES: int | None = 1 # None

# DB
MAX_DB_RESULTS: int = 5

# Maximum retry attempts if supervisor rejects response
MAX_RETRIES: int = 2


# Required slots that must be present before a booking is executed
BOOKING_REQUIRED_SLOTS: dict[str, list[str]] = {
    "book_hotel": ["hotel-bookday", "hotel-bookpeople", "hotel-bookstay"],
    "book_restaurant": ["restaurant-bookday", "restaurant-bookpeople", "restaurant-booktime"],
}


# Slot value normalization that maps LLM/user variants to canonical GT form
SLOT_VALUE_NORMALIZATION: dict[str, str] = {
    # British spelling
    "center": "centre",
    # Spacing variants
    "guest house": "guesthouse",
    # Abbreviations
    "b&b": "bed and breakfast",
    # Dontcare variants as LLM may output these when user has no preference
    "any": "dontcare",
    "doesn't matter": "dontcare",
    "do not care": "dontcare",
    "don't care": "dontcare",
    "not mentioned": "dontcare",
    "none": "dontcare",
    # Others
    "moderately priced": "moderate",
    "city centre": "centre",
    "city's centre": "centre",
    # Time variants
    "8pm": "20:00",
    "1pm": "13:00",
    "2pm": "14:00",
    "3pm": "15:00",
    "8:00 pm": "20:00",
    "this coming sunday": "sunday",
    "tonight": "today",
    # Star normalization
    "4 stars": "4",
    "3 stars": "3",
    "2 stars": "2",
    "1 star": "1",
    "5 stars": "5",
    # Internet/parking
    "free wifi": "yes",
    "free internet": "yes",
    "free parking": "yes",
}


# Map internal slot names to Tomiinek evaluator format (bookday → booking-day)
TOMIINEK_SLOT_MAP: dict[str, str] = {
    "bookday": "booking-day",
    "bookpeople": "booking-people",
    "bookstay": "booking-stay",
    "booktime": "booking-time",
}


# API-based models
API_MODELS: list[str] = [
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
]


# Open-source models
OPEN_SOURCE_MODELS: dict[str, str] = {
    "phi4_mini": "unsloth/Phi-4-mini-instruct-bnb-4bit",
    "llama32_3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3_8b": "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "llama31_8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "qwen25_7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen25_14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "qwen3_4b": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "qwen3_8b": "unsloth/Qwen3-8B-bnb-4bit",
    "qwen3_14b": "unsloth/Qwen3-14B-bnb-4bit",
    "mistral_12b": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    # "gemma4_e4b": "unsloth/gemma-4-E4B-it",
    "gemma3_12b": "unsloth/gemma-3-12b-it-bnb-4bit",

}


# Experiment 1: one model does everything (single prompt, no role split)
EXP1_CONFIGS: dict[str, dict[str, str]] = {
    "gpt": {
        "single": "gpt-4o-mini",
    },
    "haiku": {
        "single": "claude-3-haiku-20240307",
    },
    # "qwen3_8b": {
    #     "single": str(MODELS_DIR / "qwen3_8b"),
    # },
    # "gemma3_12b": {
    #     "single": str(MODELS_DIR / "gemma3_12b"),
    # },
    # "qwen25_14b": {
    #     "single": str(MODELS_DIR / "qwen25_14b"),
    # },
    # "qwen3_14b": {
    #     "single": str(MODELS_DIR / "qwen3_14b"),
    # },
}


# Experiment 2: DST and ResponseGen are separate models, zero-shot (no fine-tuning)
EXP2_CONFIGS: dict[str, dict[str, str]] = {
    "homo_gpt": {
        "dst": "gpt-4o-mini",
        "response_generator": "gpt-4o-mini",
    },
    "homo_haiku": {
        "dst": "claude-3-haiku-20240307",
        "response_generator": "claude-3-haiku-20240307",
    },
    "hetero_gpt_haiku": {
        "dst": "gpt-4o-mini",
        "response_generator": "claude-3-haiku-20240307",
    },
    "hetero_haiku_gpt": {
        "dst": "claude-3-haiku-20240307",
        "response_generator": "gpt-4o-mini",
    },

    # "homo_llama32_3b": {
    #     "dst": str(MODELS_DIR / "llama32_3b"),
    #     "response_generator": str(MODELS_DIR / "llama32_3b"),
    # },
    # "homo_llama31_8b": {
    #     "dst": str(MODELS_DIR / "llama31_8b"),
    #     "response_generator": str(MODELS_DIR / "llama31_8b"),
    # },
    # "homo_qwen3_8b": {
    #     "dst": str(MODELS_DIR / "qwen3_8b"),
    #     "response_generator": str(MODELS_DIR / "qwen3_8b"),
    # },
    # "hetero_qwen3_phi4": {
    #     "dst": str(MODELS_DIR / "qwen3_8b"),
    #     "response_generator": str(MODELS_DIR / "phi4_mini"),
    # },
    # "hetero_llama31_qwen25": {
    #     "dst": str(MODELS_DIR / "llama31_8b"),
    #     "response_generator": str(MODELS_DIR / "qwen25_7b"),
    # },
}


# Experiment 3: same pipeline as Experiment 2, but models are (Q)LoRA fine-tuned
EXP3_CONFIGS: dict[str, dict[str, str]] = {
    # "dummy_homo_lamma32_3b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "dummy/dst_lora"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "dummy/respgen_lora"),
    # },
    # "ft_homo_phi4_mini": {
    #         "dst": str(FINETUNED_MODELS_DIR / "phi4_mini_dst"),
    #         "response_generator": str(FINETUNED_MODELS_DIR / "phi4_mini_respgen"),
    #     },
    # "ft_homo_llama31_8b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "llama31_8b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "llama31_8b_respgen"),
    # },
    # "ft_homo_qwen25_7b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "qwen25_7b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "qwen25_7b_respgen"),
    # },
    # "ft_homo_qwen3_8b": {
    #         "dst": str(FINETUNED_MODELS_DIR / "qwen3_8b_dst"),
    #         "response_generator": str(FINETUNED_MODELS_DIR / "qwen3_8b_respgen"),
    #     },
    # "ft_homo_qwen3_14b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "qwen3_14b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "qwen3_14b_respgen"),
    # },
    # "ft_hetero_qwen3_llama31": {
    #         "dst": str(FINETUNED_MODELS_DIR / "qwen3_8b_dst"),
    #         "response_generator": str(FINETUNED_MODELS_DIR / "llama31_8b_respgen"),
    #     },
    # "ft_hetero_llama31_qwen25": {
    #     "dst": str(FINETUNED_MODELS_DIR / "llama31_8b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "qwen25_7b_respgen"),
    # },
    # "ft_homo_mistral_12b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "mistral_12b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "mistral_12b_respgen"),
    # },
    # "ft_homo_llama32_3b": {
    #     "dst": str(FINETUNED_MODELS_DIR / "llama32_3b_dst"),
    #     "response_generator": str(FINETUNED_MODELS_DIR / "llama32_3b_respgen"),
    # },
}


# Cost per 1000 tokens in USD (input, output) in USD (Feb 2026)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.000150, 0.000600),
    "claude-3-haiku-20240307": (0.000250, 0.001250),
    # local models are free
}

