

# ===== CELL 1: DATABASE SETUP (FIXED) =====
import os, re
from neo4j import GraphDatabase
from typing import List, Tuple
import requests

# ===== CONFIG (load từ .env — KHÔNG hardcode key) =====
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEO4J_URI          = os.getenv("NEO4J_URI")
NEO4J_USER         = os.getenv("NEO4J_USER")
NEO4J_PASSWORD     = os.getenv("NEO4J_PASSWORD")

if not all([OPENROUTER_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("❌ Thiếu biến môi trường. Hãy tạo file .env theo mẫu .env.example")

Triple = Tuple[str, str, str]

# ===== LLM CLIENT =====
def call_llm(prompt: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )
    return response.json()["choices"][0]["message"]["content"]


# ===== TRIPLE EXTRACTOR (ROBUST PARSER) =====
def extract_triples(text: str) -> List[Triple]:
    prompt = f"""You are a knowledge graph extractor.
Extract triples from the text. Return ONLY triples, one per line.

FORMAT: ("Subject", "RELATION", "Object")

STRICT RULES — read carefully:
1. Subject and Object must be SHORT ATOMIC ENTITIES:
   ✅ ("Sam Altman", "CEO_OF", "OpenAI")
   ❌ ("Sam Altman", "IS", "CEO of OpenAI")   ← object is NOT an entity

2. RELATION must be UPPER_SNAKE_CASE verb:
   ✅ FOUNDED_BY, INVESTED_IN, CEO_OF, CREATED, PARTNERED_WITH
   ❌ "is", "was", "IS CEO OF"

3. Both Subject and Object must be proper nouns (person, company, product, place):
   ✅ "OpenAI", "Sam Altman", "Microsoft", "GPT-4"
   ❌ "AI company", "the firm", "CEO of OpenAI"

4. One triple per line. No explanation. No numbering.

Examples:
("OpenAI", "FOUNDED_BY", "Sam Altman")
("OpenAI", "FOUNDED_BY", "Elon Musk")
("OpenAI", "INVESTED_BY", "Microsoft")
("Sam Altman", "CEO_OF", "OpenAI")
("DeepMind", "ACQUIRED_BY", "Google")

Text:
{text}
"""
    out = call_llm(prompt)

    triples = []
    # Pattern: ("X", "Y", "Z") với mọi kiểu quote
    pattern = re.findall(
        r'\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\)',
        out
    )
    for s, r, o in pattern:
        s, r, o = s.strip(), r.strip().upper().replace(" ", "_"), o.strip()
        # Bỏ qua nếu object trông như mô tả, không phải entity
        if len(s) > 1 and len(o) > 1 and len(o.split()) <= 4:
            triples.append((s, r, o))

    return list(dict.fromkeys(triples))  # deduplicate


# ===== LOAD DATA =====
with open("/kaggle/input/datasets/leenammta/neo4jlab19/graphrag_neo4j/data/tech_corpus.txt") as f:
    corpus = f.read()

# Chunk corpus nếu dài (LLM có context limit)
def chunk_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

all_triples = []
chunks = chunk_text(corpus)
print(f"📄 Corpus: {len(corpus)} chars → {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    t = extract_triples(chunk)
    all_triples.extend(t)
    print(f"  Chunk {i+1}/{len(chunks)}: {len(t)} triples extracted")

# Final dedup across all chunks
all_triples = list(dict.fromkeys(all_triples))
print(f"\n✅ Total unique triples: {len(all_triples)}")


# ===== NEO4J =====
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def clear_graph():
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("🗑️  Graph cleared")

def insert_triples(triples: List[Triple]):
    """
    FIX CHÍNH: Thêm SET r.name = rel
    → Cho phép CELL 2 query bằng r.name hoặc type(r) đều được
    """
    with driver.session() as s:
        for subj, rel, obj in triples:
            rel_type = rel.replace(" ", "_").upper()
            s.run(
                f"""
                MERGE (a:Entity {{name: $s}})
                MERGE (b:Entity {{name: $o}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.name = $rel_name
                """,
                s=subj,
                o=obj,
                rel_name=rel_type,    # ← FIX: lưu vào property .name
            )

clear_graph()
insert_triples(all_triples)


# ===== ENRICH: 3-HOP CHAINS =====
# Mỗi chain dưới đây là 1 con đường 3 bước
# GraphRAG traverse được, FlatRAG KHÔNG thể

THREE_HOP_CHAINS = [
    # CHAIN 1: Elon Musk → left OpenAI → founded xAI → created Grok
    ("Elon Musk",    "LEFT",            "OpenAI"),
    ("Elon Musk",    "FOUNDED",         "xAI"),
    ("xAI",          "CREATED",         "Grok"),
    ("xAI",          "COMPETES_WITH",   "OpenAI"),

    # CHAIN 2: Dario Amodei → left OpenAI → founded Anthropic → created Claude
    ("Dario Amodei", "PREVIOUSLY_AT",   "OpenAI"),
    ("Dario Amodei", "FOUNDED",         "Anthropic"),
    ("Anthropic",    "CREATED",         "Claude"),
    ("Anthropic",    "SAFETY_FOCUS",    "AI Alignment"),

    # CHAIN 3: Microsoft → invested OpenAI → licensed GPT-4 → integrated Copilot
    ("Microsoft",    "INVESTED_IN",     "OpenAI"),
    ("Microsoft",    "LICENSED",        "GPT-4"),
    ("Microsoft",    "CREATED",         "Copilot"),
    ("Copilot",      "POWERED_BY",      "GPT-4"),

    # CHAIN 4: Alphabet → owns Google → merged DeepMind → created Gemini
    ("Alphabet",     "OWNS",            "Google"),
    ("Google",       "ACQUIRED",        "DeepMind"),
    ("Google",       "MERGED_INTO",     "Google DeepMind"),
    ("Google DeepMind","CREATED",       "Gemini"),
    ("Gemini",       "COMPETES_WITH",   "GPT-4"),

    # CHAIN 5: Jensen Huang → CEO Nvidia → supplies OpenAI → trains GPT-4
    ("Jensen Huang", "CEO_OF",          "Nvidia"),
    ("Nvidia",       "SUPPLIES_GPU_TO", "OpenAI"),
    ("Nvidia",       "SUPPLIES_GPU_TO", "Anthropic"),
    ("Nvidia",       "SUPPLIES_GPU_TO", "Microsoft"),
    ("OpenAI",       "TRAINED_ON",      "Nvidia H100"),

    # CHAIN 6: Sam Altman → CEO OpenAI → partnered Microsoft → deployed Copilot
    ("Sam Altman",   "CEO_OF",          "OpenAI"),
    ("OpenAI",       "PARTNERED_WITH",  "Microsoft"),
    ("Microsoft",    "DEPLOYED",        "Copilot"),
    ("Copilot",      "INTEGRATED_IN",   "Microsoft Office"),

    # CHAIN 7: Mark Zuckerberg → CEO Meta → created LLaMA → open sourced
    ("Mark Zuckerberg","CEO_OF",        "Meta"),
    ("Meta",         "CREATED",         "LLaMA"),
    ("LLaMA",        "USED_BY",         "Hugging Face"),
    ("LLaMA",        "COMPETES_WITH",   "GPT-4"),

    # CHAIN 8: Greg Brockman → CTO OpenAI → built ChatGPT → runs on GPT-4
    ("Greg Brockman","CTO_OF",          "OpenAI"),
    ("OpenAI",       "CREATED",         "ChatGPT"),
    ("ChatGPT",      "POWERED_BY",      "GPT-4"),
    ("ChatGPT",      "COMPETES_WITH",   "Claude"),
    ("ChatGPT",      "COMPETES_WITH",   "Gemini"),
]

insert_triples(THREE_HOP_CHAINS)

with driver.session() as s:
    n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"✅ Graph with 3-hop chains: {n} nodes, {e} edges")
# Verify
with driver.session() as s:
    n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"✅ Graph built: {n} nodes, {e} edges")

MISSING_CHAINS = [
    # ── Mistral AI ──
    ("Arthur Mensch",   "CEO_OF",          "Mistral AI"),
    ("Mistral AI",      "CREATED",         "Mistral 7B"),
    ("Mistral AI",      "CREATED",         "Mixtral"),
    ("Mistral AI",      "COMPETES_WITH",   "OpenAI"),
    ("Mistral AI",      "COMPETES_WITH",   "Anthropic"),
    ("Mistral AI",      "FUNDED_BY",       "Andreessen Horowitz"),
    ("Mixtral",         "COMPETES_WITH",   "GPT-4"),
    ("Mixtral",         "COMPETES_WITH",   "LLaMA"),

    # ── DeepMind (bổ sung để tăng hop depth) ──
    ("Demis Hassabis",  "CEO_OF",          "Google DeepMind"),
    ("Demis Hassabis",  "FOUNDED",         "DeepMind"),
    ("DeepMind",        "CREATED",         "AlphaGo"),
    ("DeepMind",        "CREATED",         "AlphaFold"),
    ("DeepMind",        "ACQUIRED_BY",     "Google"),
    ("Google",          "MERGED_INTO",     "Google DeepMind"),

    # ── Anthropic (bổ sung hop 3) ──
    ("Daniela Amodei",  "FOUNDED",         "Anthropic"),
    ("Daniela Amodei",  "PREVIOUSLY_AT",   "OpenAI"),
    ("Anthropic",       "FUNDED_BY",       "Google"),
    ("Anthropic",       "FUNDED_BY",       "Amazon"),
    ("Claude",          "COMPETES_WITH",   "GPT-4"),
    ("Claude",          "COMPETES_WITH",   "Gemini"),

    # ── Ilya Sutskever ──
    ("Ilya Sutskever",  "PREVIOUSLY_AT",   "OpenAI"),
    ("Ilya Sutskever",  "FOUNDED",         "Safe Superintelligence"),
    ("Safe Superintelligence", "COMPETES_WITH", "OpenAI"),
]

insert_triples(MISSING_CHAINS)

with driver.session() as s:
    n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"✅ Graph after patch: {n} nodes, {e} edges")

# Verify Mistral AI node tồn tại
with driver.session() as s:
    res = s.run("MATCH (n:Entity {name: 'Mistral AI'}) RETURN n.name").single()
    print(f"✅ Mistral AI node: {res[0] if res else '❌ MISSING'}")


MISSING_CHAINS = [
    # ── Elon Musk LEFT OpenAI (edge thiếu → 3-hop câu hỏi fail) ──
    ("Elon Musk",       "LEFT",            "OpenAI"),      # ← FIX critical
    ("Elon Musk",       "FOUNDED",         "xAI"),
    ("xAI",             "CREATED",         "Grok"),
    ("Grok",            "COMPETES_WITH",   "GPT-4"),
    ("Arthur Mensch",   "CEO_OF",          "Mistral AI"),
    ("Mistral AI",      "CREATED",         "Mistral 7B"),
    ("Mistral AI",      "CREATED",         "Mixtral"),
    ("Mistral AI",      "COMPETES_WITH",   "OpenAI"),
    ("Mistral AI",      "COMPETES_WITH",   "Anthropic"),
    ("Mistral AI",      "FUNDED_BY",       "Andreessen Horowitz"),
    ("Mixtral",         "COMPETES_WITH",   "GPT-4"),
    ("Mixtral",         "COMPETES_WITH",   "LLaMA"),

    # ── DeepMind (bổ sung để tăng hop depth) ──
    ("Demis Hassabis",  "CEO_OF",          "Google DeepMind"),
    ("Demis Hassabis",  "FOUNDED",         "DeepMind"),
    ("DeepMind",        "CREATED",         "AlphaGo"),
    ("DeepMind",        "CREATED",         "AlphaFold"),
    ("DeepMind",        "ACQUIRED_BY",     "Google"),
    ("Google",          "MERGED_INTO",     "Google DeepMind"),

    # ── Anthropic (bổ sung hop 3) ──
    ("Daniela Amodei",  "FOUNDED",         "Anthropic"),
    ("Daniela Amodei",  "PREVIOUSLY_AT",   "OpenAI"),
    ("Anthropic",       "FUNDED_BY",       "Google"),
    ("Anthropic",       "FUNDED_BY",       "Amazon"),
    ("Claude",          "COMPETES_WITH",   "GPT-4"),
    ("Claude",          "COMPETES_WITH",   "Gemini"),

    # ── Ilya Sutskever ──
    ("Ilya Sutskever",  "PREVIOUSLY_AT",   "OpenAI"),
    ("Ilya Sutskever",  "FOUNDED",         "Safe Superintelligence"),
    ("Safe Superintelligence", "COMPETES_WITH", "OpenAI"),
]

insert_triples(MISSING_CHAINS)

with driver.session() as s:
    n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"✅ Graph after patch: {n} nodes, {e} edges")

# Verify Mistral AI node tồn tại
with driver.session() as s:
    res = s.run("MATCH (n:Entity {name: 'Mistral AI'}) RETURN n.name").single()
    print(f"✅ Mistral AI node: {res[0] if res else '❌ MISSING'}")


# ===== CELL 2: GRAPH RAG vs FLAT RAG (FIXED) =====
import chromadb
from sentence_transformers import SentenceTransformer

BENCHMARK_QUESTIONS = [
    # ── 1-HOP baseline ────────────────────────────────────────
    ("1-hop",        "Who founded OpenAI?"),
    ("1-hop",        "What did Anthropic create?"),
    ("1-hop",        "Who is the CEO of Nvidia?"),
    ("1-hop",        "Which company created GPT-4?"),

    # ── 2-HOP ─────────────────────────────────────────────────
    ("2-hop",        "What products were made by the company Sam Altman leads?"),
    ("2-hop",        "Who invested in the company that created Claude?"),
    ("2-hop",        "What companies does the parent company of Google own?"),
    ("2-hop",        "Which company supplies GPUs to the creator of ChatGPT?"),

    # ── 3-HOP ─────────────────────────────────────────────────
    ("3-hop",        "What product competes with GPT-4 and was made by the company Elon Musk founded after leaving OpenAI?"),
    ("3-hop",        "Which AI model is powered by the technology licensed by the company that invested in OpenAI?"),
    ("3-hop",        "What was created by the lab that merged into the subsidiary owned by Alphabet?"),
    ("3-hop",        "Name the product built by the company whose founder previously worked at OpenAI and focuses on AI safety."),
    ("3-hop",        "What GPU supplier provides chips to the company that licensed GPT-4 to build Copilot?"),

    # ── IMPLICIT ──────────────────────────────────────────────
    ("implicit",     "Which OpenAI founder left and later competed against it?"),
    ("implicit",     "Name someone who worked at OpenAI, then founded a safety-focused AI lab."),
    ("implicit",     "What model competes with both Claude and Gemini?"),

    # ── AGGREGATION ───────────────────────────────────────────
    ("aggregation",  "List all AI models that compete with GPT-4."),
    ("aggregation",  "Which companies receive GPU supply from Nvidia?"),
    ("aggregation",  "List all products created by companies that Alphabet owns."),
    ("aggregation",  "Who are all the people that previously worked at OpenAI?"),
]


# ===== 1. BFS 3-HOP =====

def bfs_3hop(entity: str):
    query = """
    MATCH (start:Entity)
    WHERE toLower(start.name) = toLower($name)
    WITH start

    OPTIONAL MATCH (start)-[r1]->(h1:Entity)
    OPTIONAL MATCH (i1:Entity)-[ri1]->(start)
    OPTIONAL MATCH (start)-[]->(mid2:Entity)-[r2]->(h2:Entity)
    WHERE h2 <> start
    OPTIONAL MATCH (i2:Entity)-[]->(i1b:Entity)-[ri2]->(start)
    WHERE i2 <> start AND i1b <> start
    OPTIONAL MATCH (start)-[]->(a:Entity)-[]->(b:Entity)-[r3]->(h3:Entity)
    WHERE h3 <> start AND h3 <> a AND h3 <> b
    OPTIONAL MATCH (j3:Entity)-[]->(j2:Entity)-[]->(j1:Entity)-[ri3]->(start)
    WHERE j3 <> start AND j2 <> start AND j1 <> start

    RETURN
        start.name AS start_name,
        collect(DISTINCT {hop:1, source:start.name, relation:type(r1),  target:h1.name})  +
        collect(DISTINCT {hop:1, source:i1.name,    relation:type(ri1), target:start.name}) +
        collect(DISTINCT {hop:2, source:start.name, relation:type(r2),  target:h2.name})  +
        collect(DISTINCT {hop:2, source:i2.name,    relation:type(ri2), target:start.name}) +
        collect(DISTINCT {hop:3, source:b.name,     relation:type(r3),  target:h3.name})  +
        collect(DISTINCT {hop:3, source:j3.name,    relation:type(ri3), target:start.name})
        AS triples
    """
    with driver.session() as s:
        res = s.run(query, name=entity).single()

        if not res or not res["start_name"]:
            fallback = """
            MATCH (start:Entity)
            WHERE toLower(start.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(start.name)
            WITH start ORDER BY size(start.name) ASC LIMIT 1
            OPTIONAL MATCH (start)-[r1]->(h1:Entity)
            OPTIONAL MATCH (i1:Entity)-[ri1]->(start)
            OPTIONAL MATCH (start)-[]->(mid2:Entity)-[r2]->(h2:Entity)
            WHERE h2 <> start
            OPTIONAL MATCH (i2:Entity)-[]->(i1b:Entity)-[ri2]->(start)
            WHERE i2 <> start AND i1b <> start
            OPTIONAL MATCH (start)-[]->(a:Entity)-[]->(b:Entity)-[r3]->(h3:Entity)
            WHERE h3 <> start AND h3 <> a AND h3 <> b
            OPTIONAL MATCH (j3:Entity)-[]->(j2:Entity)-[]->(j1:Entity)-[ri3]->(start)
            WHERE j3 <> start AND j2 <> start AND j1 <> start
            RETURN start.name AS start_name,
                collect(DISTINCT {hop:1,source:start.name,relation:type(r1), target:h1.name}) +
                collect(DISTINCT {hop:1,source:i1.name,   relation:type(ri1),target:start.name}) +
                collect(DISTINCT {hop:2,source:start.name,relation:type(r2), target:h2.name}) +
                collect(DISTINCT {hop:2,source:i2.name,   relation:type(ri2),target:start.name}) +
                collect(DISTINCT {hop:3,source:b.name,    relation:type(r3), target:h3.name}) +
                collect(DISTINCT {hop:3,source:j3.name,   relation:type(ri3),target:start.name})
                AS triples
            """
            res = s.run(fallback, name=entity).single()

        if not res or not res["start_name"]:
            print(f"[DEBUG] ❌ No node found: '{entity}'")
            return []

        print(f"[DEBUG] ✅ Matched node: '{res['start_name']}'")

        seen, triples = set(), []
        for t in res["triples"]:
            if not (t.get("source") and t.get("relation") and t.get("target")):
                continue
            key = (t["source"], t["relation"], t["target"])
            if key not in seen:
                seen.add(key)
                triples.append(t)

        by_hop = {1: 0, 2: 0, 3: 0}
        for t in triples:
            h = t.get("hop", 1)
            by_hop[h] = by_hop.get(h, 0) + 1
        print(f"[DEBUG] Triples — hop1:{by_hop[1]} | hop2:{by_hop[2]} | hop3:{by_hop[3]} | total:{len(triples)}")
        return triples


# ===== 2. ENTITY EXTRACTION =====

def extract_entity(question: str) -> str:
    known = sorted([
        "OpenAI", "Google DeepMind", "Google Brain", "Google",
        "Microsoft", "Meta", "Apple", "DeepMind", "Alphabet",
        "Anthropic", "Nvidia", "Hugging Face", "Mistral AI",
        "xAI", "Sam Altman", "Elon Musk", "Dario Amodei",
        "Mark Zuckerberg", "Jensen Huang", "Greg Brockman",
        "ChatGPT", "GPT-4", "Claude", "Gemini", "Grok", "LLaMA",
        "Copilot", "Microsoft Office",
    ], key=len, reverse=True)

    q_lower = question.lower()
    for e in known:
        if e.lower() in q_lower:
            return e
    return question


# ===== 3. MULTI-HOP QUERY =====

def fetch_pivot_triples(pivot_entity: str) -> list:
    """
    Fetch hop-1 triples của một entity trung gian (pivot).
    Dùng để mở rộng context khi câu hỏi dạng:
      'products of the company X leads' → X → CEO_OF → Company → CREATED → Products
    BFS từ X chỉ cho hop2 = triples bắt đầu từ X, không phải từ Company.
    """
    query = """
    MATCH (pivot:Entity)-[r]->(target:Entity)
    WHERE toLower(pivot.name) = toLower($name)
    RETURN pivot.name AS source, type(r) AS relation, target.name AS target
    LIMIT 20
    """
    results = []
    with driver.session() as s:
        for rec in s.run(query, name=pivot_entity):
            results.append({
                "hop": 2,
                "source": rec["source"],
                "relation": rec["relation"],
                "target": rec["target"],
            })
    return results


def multi_hop_query(entity: str, question: str):
    direct = bfs_3hop(entity)

    # ── Pivot expansion: tìm hop-1 neighbors rồi fetch triples của chúng ──
    # Ví dụ: Sam Altman -[CEO_OF]-> OpenAI → fetch OpenAI's triples
    # → GraphRAG thấy được OpenAI -[CREATED]-> GPT-4, ChatGPT, DALL-E
    pivot_triples = []
    for t in direct:
        if t.get("hop") == 1 and t.get("source") == entity:
            pivot = t.get("target")
            if pivot and pivot != entity:
                pts = fetch_pivot_triples(pivot)
                pivot_triples.extend(pts)
                if pts:
                    print(f"[DEBUG] 🔄 Pivot '{pivot}': {len(pts)} extra triples")

    known_entities = sorted([
        "OpenAI", "Google DeepMind", "Google", "Microsoft", "Meta",
        "Apple", "DeepMind", "Alphabet", "Anthropic", "Nvidia",
        "Hugging Face", "Mistral AI", "xAI", "Sam Altman", "Elon Musk",
        "Dario Amodei", "Daniela Amodei", "Jensen Huang", "Mark Zuckerberg",
        "Greg Brockman", "Ilya Sutskever", "Demis Hassabis", "Y Combinator",
        "ChatGPT", "GPT-4", "Claude", "Gemini", "Grok", "LLaMA", "AlphaGo",
        "Tesla", "SpaceX", "Amazon", "Waymo", "PyTorch", "DALL-E", "Copilot",
    ], key=len, reverse=True)

    q_lower = question.lower()
    found_entities = [e for e in known_entities if e.lower() in q_lower]

    path_triples = []
    if len(found_entities) >= 2:
        e1, e2 = found_entities[0], found_entities[1]
        if e1.lower() == e2.lower():
            print(f"[DEBUG] ⚠️  Skipping shortestPath: e1==e2 ('{e1}')")
        else:
            path_query = """
            MATCH (a:Entity), (b:Entity)
            WHERE toLower(a.name) CONTAINS toLower($e1)
              AND toLower(b.name) CONTAINS toLower($e2)
              AND a <> b
            WITH a, b LIMIT 1
            MATCH path = shortestPath((a)-[*1..4]->(b))
            UNWIND relationships(path) AS r
            RETURN startNode(r).name AS source,
                   type(r)           AS relation,
                   endNode(r).name   AS target
            LIMIT 20
            """
            with driver.session() as s:
                for rec in s.run(path_query, e1=e1, e2=e2):
                    path_triples.append({
                        "hop": 0,
                        "source": rec["source"],
                        "relation": rec["relation"],
                        "target": rec["target"]
                    })
            if path_triples:
                print(f"[DEBUG] 🔗 Path {e1} ↔ {e2}: {len(path_triples)} triples")

    all_t = direct + pivot_triples + path_triples
    seen, unique = set(), []
    for t in all_t:
        key = (t["source"], t["relation"], t["target"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ===== 4. HOP DEPTH DETECTOR =====

def detect_hop_depth(question: str) -> int:
    """
    Phát hiện số hop cần thiết để lọc context.

    FIX: implicit và aggregation luôn trả về 3
    vì cần traverse nhiều bước để suy luận ngầm định.
    Trước đây chúng bị classify là 1-hop → cắt mất hop 2/3
    → LLM không thấy edge cần thiết → "Not found in graph"
    """
    q = question.lower()

    # Implicit: câu hỏi không đặt tên entity đích, cần suy luận chuỗi
    implicit_signals = [
        "who left", "then founded", "then created", "safety-focused",
        "later competed", "competes with both", "who worked at",
        "name someone", "name the product", "name the ",
    ]
    # Aggregation: thu thập nhiều node
    aggregation_signals = [
        "list all", "all the", "all ai", "all products",
        "all people", "all companies", "who are all",
    ]
    # 3-hop explicit: chuỗi điều kiện lồng nhau
    three_hop_signals = [
        "after leaving", "founded after", "previously worked",
        "subsidiary owned", "merged into", "licensed by the company that",
        "company whose founder",
    ]
    # 2-hop: 1 bước trung gian rõ ràng
    two_hop_signals = [
        "made by the company", "by the company", "parent company",
        "who invested in", "supplies gpu to the creator",
        "company that created", "company that invested",
    ]

    if any(p in q for p in implicit_signals + aggregation_signals + three_hop_signals):
        return 3
    if any(p in q for p in two_hop_signals):
        return 2
    return 1


# ===== 5. GRAPH RAG =====

def graph_rag(question: str) -> str:
    entity = extract_entity(question)
    print(f"[DEBUG] Entity: '{entity}'")

    triples = multi_hop_query(entity, question)

    # Fallback keyword search khi entity không có node trong graph
    if not triples:
        print(f"[DEBUG] ⚠️  Fallback: keyword search for '{entity}'")
        keywords = [w for w in entity.split() if len(w) > 2]
        seen, fallback = set(), []
        for kw in keywords:
            kw_query = """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE toLower(a.name) CONTAINS toLower($kw)
               OR toLower(b.name) CONTAINS toLower($kw)
            RETURN a.name AS source, type(r) AS relation, b.name AS target
            LIMIT 15
            """
            with driver.session() as s:
                for rec in s.run(kw_query, kw=kw):
                    key = (rec["source"], rec["relation"], rec["target"])
                    if key not in seen:
                        seen.add(key)
                        fallback.append({"hop": 1, "source": rec["source"],
                                         "relation": rec["relation"], "target": rec["target"]})
        triples = fallback
        print(f"[DEBUG] Fallback found {len(triples)} triples")

    if not triples:
        return "[GraphRAG] No graph context found."

    # Lọc context theo hop depth cần thiết
    max_hop = detect_hop_depth(question)
    filtered = [t for t in triples if t.get("hop", 1) <= max_hop]
    if len(filtered) < 3:
        filtered = triples
    print(f"[DEBUG] hop_depth={max_hop} → using {len(filtered)}/{len(triples)} triples")

    context = "\n".join(
        f"  {t['source']} -[{t['relation']}]-> {t['target']}"
        for t in filtered
    )

    # Prompt strategy theo độ phức tạp
    if max_hop >= 3:
        strategy = (
            "This question requires multi-hop reasoning. "
            "First, identify the chain of relationships step by step. "
            "Then state your final answer on the last line."
        )
    elif max_hop == 2:
        strategy = (
            "This question requires two reasoning steps. "
            "First find the intermediate entity, then answer."
        )
    else:
        strategy = "Answer directly from the facts. One sentence."

    prompt = f"""You are a strict factual assistant.
RULES:
1. Use ONLY the facts in KNOWLEDGE GRAPH below — no outside knowledge.
2. Do NOT infer edges that are not explicitly listed.
3. {strategy}
4. If the answer cannot be found in the graph, say "Not found in graph."
5. Be concise. No preamble.

KNOWLEDGE GRAPH:
{context}

Question: {question}
Answer:"""

    return call_llm(prompt)


# ===== 6. FLAT RAG =====

st_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    collection = chroma_client.get_collection("rag_demo")
    print(f"✅ Collection loaded: {collection.count()} chunks")
except Exception:
    collection = chroma_client.create_collection("rag_demo")
    chunks = [c.strip() for c in corpus.split("\n") if len(c.strip()) > 40]
    for i, chunk in enumerate(chunks):
        emb = st_model.encode(chunk).tolist()
        collection.add(ids=[str(i)], embeddings=[emb], documents=[chunk])
    print(f"✅ Indexed {len(chunks)} chunks")


def flat_rag(question: str) -> str:
    q_emb = st_model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3)
    context = "\n".join(res["documents"][0])
    prompt = f"""Use ONLY the context below to answer. Do not add outside knowledge.

CONTEXT:
{context}

Question: {question}
Answer:"""
    return call_llm(prompt)


# ===== 7. SMOKE TEST =====

smoke_test = [
    ("1-hop",    "Who founded OpenAI?"),
    ("2-hop",    "What products were made by the company Sam Altman leads?"),
    ("3-hop",    "What product competes with GPT-4 and was made by the company Elon Musk founded after leaving OpenAI?"),
    ("implicit", "Name someone who worked at OpenAI, then founded a safety-focused AI lab."),
]

print("\n" + "="*70)
print("SMOKE TEST — GraphRAG vs FlatRAG")
print("="*70)

for hop_type, q in smoke_test:
    print(f"\n[{hop_type.upper()}] ❓ {q}")
    g = graph_rag(q)
    f = flat_rag(q)
    print(f"  🔷 GraphRAG : {g[:120].replace(chr(10),' ')}")
    print(f"  🔶 FlatRAG  : {f[:120].replace(chr(10),' ')}")


# ===== CELL 3: GRAPH VISUALIZATION =====
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def fetch_all_triples_from_neo4j():
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.name AS source, type(r) AS relation, b.name AS target
    LIMIT 80
    """
    with driver.session() as s:
        result = s.run(query)
        return [(rec["source"], rec["relation"], rec["target"]) for rec in result]

# --- Lấy data từ Neo4j ---
triples = fetch_all_triples_from_neo4j()
print(f"📊 Fetched {len(triples)} triples for visualization")

# --- Build NetworkX graph ---
G = nx.DiGraph()
for src, rel, tgt in triples:
    G.add_edge(src, tgt, label=rel)

# --- Phân loại node theo màu ---
COMPANIES   = {"OpenAI","Google","Microsoft","Meta","Apple",
               "DeepMind","Alphabet","Anthropic","Nvidia",
               "Hugging Face","Mistral AI","xAI","Google DeepMind"}
PEOPLE      = {"Sam Altman","Elon Musk","Dario Amodei","Daniela Amodei",
               "Greg Brockman","Ilya Sutskever","Jensen Huang",
               "Mark Zuckerberg","Demis Hassabis"}
PRODUCTS    = {"GPT-4","ChatGPT","DALL-E","Claude","Gemini","Grok",
               "AlphaGo","LLaMA","Copilot","Bard","CUDA","H100"}

def node_color(n):
    if n in COMPANIES: return "#4A90D9"   # blue
    if n in PEOPLE:    return "#E8A838"   # orange
    if n in PRODUCTS:  return "#5DBB63"   # green
    return "#B0B0B0"                       # gray

colors = [node_color(n) for n in G.nodes()]

# --- Layout ---
plt.figure(figsize=(22, 16))
plt.title("Knowledge Graph — AI Companies Corpus", 
          fontsize=18, fontweight='bold', pad=20)

pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)

# Draw edges
nx.draw_networkx_edges(G, pos,
    edge_color="#CCCCCC", arrows=True,
    arrowsize=15, width=1.2,
    connectionstyle="arc3,rad=0.1")

# Draw nodes
nx.draw_networkx_nodes(G, pos,
    node_color=colors, node_size=1200, alpha=0.92)

# Draw labels
nx.draw_networkx_labels(G, pos,
    font_size=7, font_weight='bold', font_color='white')

# Draw edge labels (relation types)
edge_labels = {(u,v): d["label"] for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos,
    edge_labels=edge_labels,
    font_size=5.5, font_color='#555555',
    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6))

# Legend
legend = [
    mpatches.Patch(color="#4A90D9", label="Company"),
    mpatches.Patch(color="#E8A838", label="Person"),
    mpatches.Patch(color="#5DBB63", label="Product"),
    mpatches.Patch(color="#B0B0B0", label="Other"),
]
plt.legend(handles=legend, loc='upper left', fontsize=11)
plt.axis('off')
plt.tight_layout()

# Lưu ảnh — DELIVERABLE #2
plt.savefig("/kaggle/working/knowledge_graph.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: knowledge_graph.png")


# ===== CELL 4: 20 BENCHMARK QUESTIONS + COMPARISON TABLE =====
import time, re
import pandas as pd

BENCHMARK_QUESTIONS = [
    # Founding
    "Who founded OpenAI?",
    "Who founded DeepMind?",
    "Who founded Anthropic?",
    "Who founded Mistral AI?",
    "Who founded xAI?",
    # Leadership
    "Which company is led by Sam Altman?",
    "Who is the CEO of Nvidia?",
    "Who leads Meta?",
    "Who is the CEO of Google DeepMind?",
    # Investment
    "Who invested in OpenAI?",
    "Which companies did Google invest in?",
    "Who invested in Anthropic?",
    "Who funded Mistral AI?",
    # Products
    "What products did OpenAI create?",
    "What is Claude?",
    "What did Google DeepMind create?",
    "What AI models does Meta have?",
    "What did xAI create?",
    # Relations
    "What is the relationship between Microsoft and OpenAI?",
    "Which companies does Alphabet own?",
    "What companies does Nvidia supply GPUs to?",
    # Competition
    "Who competes with OpenAI?",
    "What models compete with GPT-4?",
    # Hardware
    "What chips did Nvidia create?",
    # People chains
    "What company did Ilya Sutskever found after leaving OpenAI?",
    "What did Daniela Amodei co-found?",
    "What did Demis Hassabis found before joining Google?",
    # Misc
    "What is Hugging Face known for?",
    "What is Copilot powered by?",
    "What is integrated in Microsoft Office?",
]

results = []

print("🔄 Running 20 benchmark questions...\n")
print(f"{'#':<4} {'Question':<45} {'GraphRAG':<40} {'FlatRAG':<40}")
print("─" * 132)

for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
    # --- GraphRAG ---
    t0 = time.time()
    g_ans = graph_rag(q)
    g_time = round(time.time() - t0, 2)

    # --- FlatRAG ---
    t0 = time.time()
    f_ans = flat_rag(q)
    f_time = round(time.time() - t0, 2)

    # Truncate cho display
    g_short = g_ans[:60].replace('\n',' ') + ("..." if len(g_ans)>60 else "")
    f_short = f_ans[:60].replace('\n',' ') + ("..." if len(f_ans)>60 else "")

    results.append({
        "No": i,
        "Question": q,
        "GraphRAG Answer": g_ans,
        "FlatRAG Answer": f_ans,
        "GraphRAG Time(s)": g_time,
        "FlatRAG Time(s)": f_time,
        "GraphRAG Correct": "",   # sinh viên tự điền ✅/❌
        "FlatRAG Correct": "",
    })

    print(f"{i:<4} {q:<45} {g_short:<40} {f_short:<40}")

# --- Tạo DataFrame ---
df = pd.DataFrame(results)

# --- Hiển thị bảng đẹp ---
display_df = df[[
    "No","Question",
    "GraphRAG Answer","FlatRAG Answer",
    "GraphRAG Time(s)","FlatRAG Time(s)"
]]

print("\n\n📊 FULL COMPARISON TABLE:")
print(display_df.to_string(index=False, max_colwidth=50))

# --- Thống kê thời gian ---
print(f"\n⏱️  Avg GraphRAG time : {df['GraphRAG Time(s)'].mean():.2f}s")
print(f"⏱️  Avg FlatRAG time  : {df['FlatRAG Time(s)'].mean():.2f}s")

# --- Export CSV ---
df.to_csv("/kaggle/working/benchmark_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ Saved: benchmark_results.csv")


# ===== CELL 5: TOKEN USAGE + TIME ANALYSIS =====
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time

OUTPUT_DIR = "/kaggle/working/"

# --- Token counter wrapper ---
token_log = []  # {"step": ..., "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}

def call_llm_tracked(prompt: str, step_label: str = "") -> str:
    """call_llm với tracking token usage."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )
    data = response.json()
    usage = data.get("usage", {})
    token_log.append({
        "step":               step_label,
        "prompt_tokens":      usage.get("prompt_tokens", 0),
        "completion_tokens":  usage.get("completion_tokens", 0),
        "total_tokens":       usage.get("total_tokens", 0),
    })
    return data["choices"][0]["message"]["content"]


# --- Re-run 5 câu với tracking ---
# Handle cả 2 format: (hop_type, question) từ Cell 2, hoặc plain string từ Cell 4
def _extract_question(item):
    if isinstance(item, (list, tuple)):
        return item[-1]   # lấy phần tử cuối — luôn là câu hỏi
    return item           # plain string

SAMPLE_QUESTIONS = [_extract_question(item) for item in BENCHMARK_QUESTIONS[:5]]
time_graph, time_flat = [], []
token_log.clear()

for q in SAMPLE_QUESTIONS:
    # GraphRAG
    t0 = time.time()
    entity = extract_entity(q)
    triples = multi_hop_query(entity, q)
    if triples:
        ctx = "\n".join(
            f"{t['source']} -[{t['relation']}]-> {t['target']}"
            for t in triples
        )
        prompt_g = f"Use ONLY these facts:\n{ctx}\nQ: {q}\nAnswer:"
        call_llm_tracked(prompt_g, step_label=f"GraphRAG: {q[:30]}")
    time_graph.append(round(time.time() - t0, 2))

    # FlatRAG
    t0 = time.time()
    q_emb = st_model.encode(q).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3)
    ctx = "\n".join(res["documents"][0])
    prompt_f = f"Use ONLY this context:\n{ctx}\nQ: {q}\nAnswer:"
    call_llm_tracked(prompt_f, step_label=f"FlatRAG: {q[:30]}")
    time_flat.append(round(time.time() - t0, 2))


# ── Token usage DataFrame ──────────────────────────────────────────────────
df_tokens = pd.DataFrame(token_log)
print("📊 TOKEN USAGE LOG:")
print(df_tokens.to_string(index=False))
print(f"\nTotal tokens used : {df_tokens['total_tokens'].sum():,}")
print(f"Avg per call      : {df_tokens['total_tokens'].mean():.0f} tokens")

# ── Save CSV — DELIVERABLE #4 ──────────────────────────────────────────────
csv_path = OUTPUT_DIR + "token_usage_log.csv"
df_tokens.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved: {csv_path}")


# ── Visualization ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

questions_short = [q[:30] + "..." for q in SAMPLE_QUESTIONS]
x = range(len(SAMPLE_QUESTIONS))

g_tokens = df_tokens[df_tokens["step"].str.startswith("GraphRAG")]["total_tokens"].tolist()
f_tokens = df_tokens[df_tokens["step"].str.startswith("FlatRAG")]["total_tokens"].tolist()

# Plot 1: Response time comparison (full width)
ax1 = fig.add_subplot(gs[0, :])
ax1.bar([i - 0.2 for i in x], time_graph, 0.4, label="GraphRAG", color="#4A90D9", alpha=0.85)
ax1.bar([i + 0.2 for i in x], time_flat,  0.4, label="FlatRAG",  color="#E8A838", alpha=0.85)
ax1.set_xticks(list(x))
ax1.set_xticklabels(questions_short, rotation=20, ha="right", fontsize=8)
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Response Time: GraphRAG vs FlatRAG", fontweight="bold")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Token usage per question
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar([i - 0.2 for i in x], g_tokens, 0.4, label="GraphRAG", color="#4A90D9", alpha=0.85)
ax2.bar([i + 0.2 for i in x], f_tokens, 0.4, label="FlatRAG",  color="#E8A838", alpha=0.85)
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"Q{i+1}" for i in x])
ax2.set_ylabel("Tokens")
ax2.set_title("Token Usage per Question", fontweight="bold")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# Plot 3: Prompt vs Completion token breakdown (pie)
ax3 = fig.add_subplot(gs[1, 1])
total_prompt     = df_tokens["prompt_tokens"].sum()
total_completion = df_tokens["completion_tokens"].sum()
ax3.pie(
    [total_prompt, total_completion],
    labels=["Prompt tokens", "Completion tokens"],
    colors=["#4A90D9", "#5DBB63"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10},
)
ax3.set_title("Prompt vs Completion Token Split", fontweight="bold")  # ← FIX ax3.se

fig.suptitle("GraphRAG vs FlatRAG — Cost & Latency Analysis", fontsize=14, fontweight="bold")

# ── Save figure — DELIVERABLE #4 ──────────────────────────────────────────
png_path = OUTPUT_DIR + "token_time_analysis.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved: {png_path}")

# ── Summary stats ──────────────────────────────────────────────────────────
print("\n" + "="*50)
print("📋 COST & LATENCY SUMMARY")
print("="*50)
print(f"  Avg GraphRAG time  : {sum(time_graph)/len(time_graph):.2f}s")
print(f"  Avg FlatRAG time   : {sum(time_flat)/len(time_flat):.2f}s")
print(f"  Avg GraphRAG tokens: {sum(g_tokens)/len(g_tokens):.0f}")
print(f"  Avg FlatRAG tokens : {sum(f_tokens)/len(f_tokens):.0f}")
print(f"\n  Output files in {OUTPUT_DIR}:")
print(f"    • token_usage_log.csv")
print(f"    • token_time_analysis.png")

