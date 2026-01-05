from pathlib import Path

SPLITS = ["train", "val", "test", "micro_dev"]
BASE_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DATA_DIR.parent

DATASET_SPECS = {
    "english_qa": ("english_qa", "en"),
    "arabic_qa": ("arabic_qa", "ar"),
    "chinese_qa": ("chinese_qa", "zh"),
    "english_summary": ("english_summary", "en"),
    "arabic_summary": ("arabic_summary", "ar"),
    "chinese_summary": ("chinese_summary", "zh"),
}

DATASET_LISTS = list(DATASET_SPECS.keys())

DATASET_FILES = {
    "english_qa": {
        "train": [str(BASE_DATA_DIR / "english_qa" / "input" / "en.train.csv")],
        "val": [str(BASE_DATA_DIR / "english_qa" / "input" / "en.val.csv")],
        "test": [str(BASE_DATA_DIR / "english_qa" / "input" / "en.test.csv")],
    },
    "arabic_qa": {
        "train": [str(BASE_DATA_DIR / "arabic_qa" / "input" / "ar.train.csv")],
        "val": [str(BASE_DATA_DIR / "arabic_qa" / "input" / "ar.val.csv")],
        "test": [str(BASE_DATA_DIR / "arabic_qa" / "input" / "ar.test.csv")],
        "micro_dev": [str(BASE_DATA_DIR / "arabic_qa" / "input" / "ar.micro_dev.csv")],
    },
    "chinese_qa": {
        "train": [str(BASE_DATA_DIR / "chinese_qa" / "input" / "zh.train.csv")],
        "val": [str(BASE_DATA_DIR / "chinese_qa" / "input" / "zh.val.csv")],
        "test": [str(BASE_DATA_DIR / "chinese_qa" / "input" / "zh.test.csv")],
        "micro_dev": [str(BASE_DATA_DIR / "chinese_qa" / "input" / "zh.micro_dev.csv")],
    },
    "english_summary": {
        "train": [str(BASE_DATA_DIR / "english_summary" / "input" / "en.train.csv")],
        "val": [str(BASE_DATA_DIR / "english_summary" / "input" / "en.val.csv")],
        "test": [str(BASE_DATA_DIR / "english_summary" / "input" / "en.test.csv")],
    },
    "arabic_summary": {
        "train": [str(BASE_DATA_DIR / "arabic_summary" / "input" / "ar.train.csv")],
        "val": [str(BASE_DATA_DIR / "arabic_summary" / "input" / "ar.val.csv")],
        "test": [str(BASE_DATA_DIR / "arabic_summary" / "input" / "ar.test.csv")],
        "micro_dev": [str(BASE_DATA_DIR / "arabic_summary" / "input" / "ar.micro_dev.csv")],
    },
    "chinese_summary": {
        "train": [str(BASE_DATA_DIR / "chinese_summary" / "input" / "zh.train.csv")],
        "val": [str(BASE_DATA_DIR / "chinese_summary" / "input" / "zh.val.csv")],
        "test": [str(BASE_DATA_DIR / "chinese_summary" / "input" / "zh.test.csv")],
        "micro_dev": [str(BASE_DATA_DIR / "chinese_summary" / "input" / "zh.micro_dev.csv")],
    },
}

LORA_PATH_DICT = {
    "english_qa": str(REPO_ROOT / "merge/output/english_qa_adapter"),
    "arabic_qa": str(REPO_ROOT / "merge/output/arabic_qa_adapter"),
    "chinese_qa": str(REPO_ROOT / "merge/output/chinese_qa_adapter"),
    "english_summary": str(REPO_ROOT / "merge/output/english_summary_adapter"),
    "arabic_summary": str(REPO_ROOT / "merge/output/arabic_summary_adapter"),
    "chinese_summary": str(REPO_ROOT / "merge/output/chinese_summary_adapter"),
}

PROMPT_TEMPLATES = {
    "arabic_summary": "لخّص الخبر العربي التالي.\nالمقال:{article}\nالملخّص:",
    "english_summary": "Summarize the following English news article.\nArticle:{article}\nSummary:",
    "chinese_summary": "请用中文概括下面的新闻报道。\n文章:{article}\n摘要:",
    "english_qa": "Answer the question based on the context.\nContext:{context}\nQuestion:{question}\nAnswer:",
    "arabic_qa": "أجب على السؤال بناءً على النص التالي.\nالنص:{context}\nالسؤال:{question}\nالإجابة:",
    "chinese_qa": "请根据下列内容回答问题。\n内容:{context}\n问题:{question}\n答案:",
}

SUMMARY_WORDS = {
    "arabic_summary": "الملخّص:",
    "english_summary": "Summary:",
    "chinese_summary": "摘要:",
    "english_qa": "Answer:",
    "arabic_qa": "الإجابة:",
    "chinese_qa": "答案:",
}

MAX_TOKENS_DICT = {
    "arabic_summary": {"max_new_tokens": 128, "max_io_tokens": 2200},
    "english_summary": {"max_new_tokens": 96, "max_io_tokens": 2200},
    "chinese_summary": {"max_new_tokens": 96, "max_io_tokens": 2200},
    "english_qa": {"max_new_tokens": 64, "max_io_tokens": 2200},
    "arabic_qa": {"max_new_tokens": 64, "max_io_tokens": 2200},
    "chinese_qa": {"max_new_tokens": 64, "max_io_tokens": 2200},
}
