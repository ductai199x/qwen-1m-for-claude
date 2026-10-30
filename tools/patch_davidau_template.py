"""Make DavidAU's Qwen3.8 chat template ("Custom Reasoning // Instruction System") usable behind Claude Code.

Applies the same three fixes as this repo's chat_template.jinja, leaving everything else (the {REASON:mode} system,
its help text and attribution check) untouched:
  1. reasoning_effort: Claude Code sends "high", and vLLM can pass None; both made the template raise. high/max map to
     xhigh, minimal to low, anything else to the template's defaults. A {REASON:...} tag in a user message still wins.
  2. unknown content blocks are skipped instead of raising.
  3. non-leading system messages (Claude Code's per-turn reminders) render in place instead of raising, so vLLM does not
     merge them into the leading system prompt (which breaks the prefix cache).

The template is copyright DavidAU ("contact me for permissions / mods / use"); the output stays next to the model
and is not committed. Usage: python patch_davidau_template.py IN.jinja OUT.jinja
"""
import sys

src, dst = sys.argv[1:3]
s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"expected {count}x: {old[:70]!r} (found {s.count(old)})"
    s = s.replace(old, new)


# 1. effort normalisation, before the {REASON:...} override so a tag still takes precedence
sub("{%- if store.reasoning_effort and l_car=='8'  %}",
    "{#- Patched: map OpenAI/Anthropic effort levels (Claude Code sends \"high\"; vLLM may pass None) onto this\n"
    "    template's modes instead of raising. #}\n"
    "{%- if reasoning_effort is defined and reasoning_effort %}\n"
    "  {%- if reasoning_effort in ('high', 'max') %}{%- set reasoning_effort = 'xhigh' %}\n"
    "  {%- elif reasoning_effort == 'minimal' %}{%- set reasoning_effort = 'low' %}\n"
    "  {%- elif reasoning_effort not in reason_mods %}{%- set reasoning_effort = '' %}{%- endif %}\n"
    "{%- endif %}\n"
    "{%- if store.reasoning_effort and l_car=='8'  %}")
sub("{%- set resolved_reasoning_effort = reasoning_effort|default(default_effort) %}",
    "{%- set resolved_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort else default_effort %}")
sub("{%- set resolved_reasoning_effort = reasoning_effort|default(default_effort_instruct) %}",
    "{%- set resolved_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort else default_effort_instruct %}")
# (the two "Unexpected reasoning effort" raises stay: after normalisation they can no longer fire)

# 2. unknown content blocks
sub("{{- raise_exception('Unexpected item type in content.') }}", "{#- Patched: skip unknown content block types. #}")

# 3. non-leading system messages render in place
sub("""        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}""",
    """        {#- Patched: render non-leading system messages (e.g. Claude Code's per-turn reminders) in place, so vLLM
            doesn't merge them into the leading system prompt and invalidate the prefix cache. #}
        {%- if not loop.first %}
            {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
        {%- endif %}""")

open(dst, "w").write(s)
print(f"wrote {dst}")
