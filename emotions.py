"""Emotion scheme shared by proposition_tracker.py, build_viewer.py and analyze_corpus.py (single source of truth)."""

# Emotions of protest after Jasper (1998), "The Emotions of Protest: Affective and Reactive Emotions in and
# Around Social Movements", Sociological Forum, Table I ("Some emotions potentially relevant to protest").
# The 17 categories and their grouping (affective / reactive / moods and others in between) follow the table.
# 'neutral' is NOT part of Jasper's table; it is only for paragraphs that express no emotion.
# Each entry: key -> (Jasper's group, words from the table that belong to the category, short gloss for the prompt;
# the glosses are paraphrases of the remarks in the table, not quotations).
JASPER_EMOTIONS = {
    "hatred_hostility": ("affective", ("hatred", "hostility", "loathing"), "hatred, hostility or loathing towards opponents"),
    "love": ("affective", ("love",), "love or attachment to people in the movement"),
    "solidarity_loyalty": ("affective", ("solidarity", "loyalty"), "solidarity, loyalty, positive feelings towards the group"),
    "suspicion_paranoia": ("affective", ("suspicion", "paranoia"), "suspicion or paranoia about opponents' motives"),
    "trust_respect": ("affective", ("trust", "respect"), "trust or respect towards others (allies, institutions)"),
    "anger": ("reactive", ("anger", "rage"), "anger or rage"),
    "grief_sorrow": ("reactive", ("grief", "loss", "sorrow"), "grief, loss or sorrow"),
    "outrage_indignation": ("reactive", ("outrage", "indignation"), "outrage or indignation directed at a blamed target"),
    "shame": ("reactive", ("shame",), "shame"),
    "compassion_sympathy": ("mood", ("compassion", "sympathy", "pity"), "compassion, sympathy or pity for the plight of others"),
    "cynicism_depression": ("mood", ("cynicism", "depression"), "cynicism or depression; hopes for change dampened"),
    "defiance": ("mood", ("defiance",), "defiance; a stance of resistance"),
    "enthusiasm_pride": ("mood", ("enthusiasm", "pride"), "enthusiasm for the cause, pride in the collective identity"),
    "envy_resentment": ("mood", ("envy", "resentment"), "envy or resentment"),
    "fear_dread": ("mood", ("fear", "dread"), "fear or dread, a sense of threat"),
    "joy_hope": ("mood", ("joy", "hope"), "joy, empowerment, hope; anticipation of a better future"),
    "resignation": ("mood", ("resignation",), "resignation; a sense that change is not possible"),
}
EMOTIONS = list(JASPER_EMOTIONS) + ["neutral"]
EMOTION_GROUP = {**{k: v[0] for k, v in JASPER_EMOTIONS.items()}, "neutral": "neutral"}
# Words an LLM may return instead of the key. 'sadness' is not in Jasper's table; it is mapped to grief/sorrow.
EMOTION_ALIASES = {**{w: k for k, (_g, words, _t) in JASPER_EMOTIONS.items() for w in (*words, k)}, "sadness": "grief_sorrow"}

# DIRECTION is used ONLY to decide when an emotion change counts as a "flip" (flag emotion_valence_flip).
# Jasper's table gives no valence. This mapping is the analyst's reading of the remarks in the table:
#   +1 = described as building or encouraging protest (love, solidarity, trust, outrage, compassion, defiance,
#        enthusiasm/pride, joy/hope)
#   -1 = described as dampening protest or the sense that change is possible (cynicism/depression, resignation)
#    0 = mixed or channelled elsewhere (anger, fear, shame, grief, hatred, suspicion, envy; can mobilise or paralyse)
# A flip therefore means: encouraging <-> dampening. Edit to taste, and report the mapping.
EMOTION_DIRECTION = {
    "love": 1, "solidarity_loyalty": 1, "trust_respect": 1, "outrage_indignation": 1, "compassion_sympathy": 1,
    "defiance": 1, "enthusiasm_pride": 1, "joy_hope": 1,
    "cynicism_depression": -1, "resignation": -1,
    "hatred_hostility": 0, "suspicion_paranoia": 0, "anger": 0, "grief_sorrow": 0, "shame": 0,
    "envy_resentment": 0, "fear_dread": 0, "neutral": 0,
}
