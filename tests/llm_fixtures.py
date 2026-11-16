"""A tiny, randomly initialised chat model built offline for smoke tests of
the Phase 2 code (no model download, no GPU). Its outputs are meaningless;
only the plumbing is under test."""

from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tiny_model(texts: list[str], out_dir: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    raw = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=special,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  unk_token="<|endoftext|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
