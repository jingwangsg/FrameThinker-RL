from .default import apply_message_template as apply_message_template_default
from .framethinker_default import (
    apply_message_template as apply_message_template_framethinker_default,
)

from .framethinker_add_zoomin import (
    apply_message_template as apply_message_template_framethinker_add_zoomin,
)

MESSAGE_TEMPLATES = {
    "default": apply_message_template_default,
    "framethinker_default": apply_message_template_framethinker_default,
    "framethinker_add_zoomin": apply_message_template_framethinker_add_zoomin,
}

def get_message_template(name: str):
    assert name in MESSAGE_TEMPLATES, f"Message template {name} not found in {MESSAGE_TEMPLATES.keys()}"
    return MESSAGE_TEMPLATES[name]