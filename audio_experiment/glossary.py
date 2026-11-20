"""Character names. VERIFIED spellings come from the user; the rest are proposals by Qwen and are marked unverified."""

# DB title -> verified Japanese spellings. `names` are used to score "addressed by name" contradictions
# (given names and stand-alone family names only; 梓川 is shared by two siblings and left out on purpose).
VERIFIED = {
    'Sakuta Azusagawa': {'ja': '梓川咲太', 'names': ['咲太', 'サクタ', 'さくた']},
    'Mai Sakurajima': {'ja': '桜島麻衣', 'names': ['麻衣', 'マイ', '桜島']},
    'Kaede Azusagawa': {'ja': '梓川かえで', 'names': ['かえで', 'カエデ', '楓']},
    'Rio Futaba': {'ja': '双葉理央', 'names': ['理央', 'リオ', '双葉']},
    'Yuuma Kunimi': {'ja': '国見佑真', 'names': ['佑真', 'ユウマ', '国見']},
    'Tomoe Koga': {'ja': '古賀朋絵', 'names': ['朋絵', 'トモエ', '古賀']},
    'Shoko Makinohara': {'ja': '牧之原翔子', 'names': ['翔子', 'ショウコ', '牧之原']},
    'Saki Kamisato': {'ja': '上里沙希', 'names': ['沙希', 'サキ', '上里']},
    'Fumika Nanjou': {'ja': '南条文香', 'names': ['文香', 'フミカ', '南条']},
}
# Round 2: the verified spellings split into family / given name with their kana readings (standard readings of these names).
# Used by names.py (kana -> kanji, and the Qwen name pass, which never saw the Japanese spellings in round 1).
PARTS = {
    'Sakuta Azusagawa': [('梓川', 'あずさがわ'), ('咲太', 'さくた')],
    'Mai Sakurajima': [('桜島', 'さくらじま'), ('麻衣', 'まい')],
    'Kaede Azusagawa': [('梓川', 'あずさがわ'), ('かえで', 'かえで')],
    'Rio Futaba': [('双葉', 'ふたば'), ('理央', 'りお')],
    'Yuuma Kunimi': [('国見', 'くにみ'), ('佑真', 'ゆうま')],
    'Tomoe Koga': [('古賀', 'こが'), ('朋絵', 'ともえ')],
    'Shoko Makinohara': [('牧之原', 'まきのはら'), ('翔子', 'しょうこ')],
    'Saki Kamisato': [('上里', 'かみさと'), ('沙希', 'さき')],
    'Fumika Nanjou': [('南条', 'なんじょう'), ('文香', 'ふみか')],
}
HONORIFICS =['さん', 'くん', '君', 'ちゃん', '先輩', 'せんぱい', 'さま', '様', '先生']

# first-person pronouns grouped by reading class (orthographic variants of one pronoun are one class)
PRONOUNS = {
    'ore': ['俺', 'おれ', 'オレ'],
    'boku': ['僕', 'ぼく', 'ボク'],
    'watashi': ['私', 'わたし', 'ワタシ'],
    'atashi': ['あたし', 'アタシ'],
    'washi': ['わし'],
}


def verified_context_names(titles):
    """Japanese spellings of the verified characters among `titles` (db titles)."""
    return [VERIFIED[t]['ja'] for t in titles if t in VERIFIED]


def addressed_patterns(character):
    """Regex source that finds this character addressed by name + honorific, or None for characters without verified names."""
    entry = VERIFIED.get(character)
    if not entry:
        return None
    names = '|'.join(sorted(entry['names'], key=len, reverse=True))
    honorifics = '|'.join(HONORIFICS)
    return f'(?:{names})(?:{honorifics})'
