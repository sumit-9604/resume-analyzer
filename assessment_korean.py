import math

SKILL_ALIASES = {
    # Languages
    "python":                "python",
    "pyhton":                "python",
    "java":                  "java",
    "javascript":            "javascript",
    "javascrpit":            "javascript",
    "js":                    "javascript",
    "typescript":            "typescript",
    "typescrpit":            "typescript",
    "c++":                   "cpp",
    "cpp":                   "cpp",
    "r":                     "r",
    "kotlin":                "kotlin",
    # ML / Data
    "machinelearning":       "machine_learning",
    "machine learning":      "machine_learning",
    "ml":                    "machine_learning",
    "sklearn":               "machine_learning",
    "deeplearning":          "deep_learning",
    "deep learning":         "deep_learning",
    "deep-learning":         "deep_learning",
    "tensorflow":            "tensorflow",
    "pytorch":               "pytorch",
    "keras":                 "keras",
    "nlp":                   "nlp",
    "bert":                  "bert",
    "xgboost":               "xgboost",
    "feature engineering":   "feature_engineering",
    "statistics":            "statistics",
    "stats":                 "statistics",
    "regression":            "regression",
    "clustering":            "clustering",
    "data-viz":              "data_visualization",
    "data visualization":    "data_visualization",
    "data viz":              "data_visualization",
    "matplotlib":            "data_visualization",
    "tableau":               "data_visualization",
    "power-bi":              "data_visualization",
    "power bi":              "data_visualization",
    "powerbi":               "data_visualization",
    "pandas":                "pandas",
    "numpy":                 "numpy",
    # Web — Frontend
    "react":                 "react",
    "reacts":                "react",
    "reactjs":               "react",
    "vue":                   "vue",
    "vue.js":                "vue",
    "vuejs":                 "vue",
    "redux":                 "redux",
    "tailwind":              "tailwind",
    "html/css":              "html_css",
    "html css":              "html_css",
    "html":                  "html_css",
    "css":                   "html_css",
    "jest":                  "jest",
    "graphql":               "graphql",
    # Web — Backend
    "node.js":               "nodejs",
    "nodejs":                "nodejs",
    "node js":               "nodejs",
    "flask":                 "flask",
    "spring boot":           "spring_boot",
    "springboot":            "spring_boot",
    "rest api":              "rest_api",
    "rest":                  "rest_api",
    "restapi":               "rest_api",
    "microservices":         "microservices",
    # Databases
    "sql":                   "sql",
    "mysql":                 "mysql",
    "mysq":                  "mysql",
    "postgresql":            "postgresql",
    "postgres":              "postgresql",
    "mongodb":               "mongodb",
    "redis":                 "redis",
    # DevOps / Cloud
    "docker":                "docker",
    "kubernetes":            "kubernetes",
    "kubernates":            "kubernetes",
    "k8s":                   "kubernetes",
    "ci/cd":                 "ci_cd",
    "cicd":                  "ci_cd",
    "ci cd":                 "ci_cd",
    "aws":                   "aws",
    # Mobile
    "android":               "android",
    "firebase":              "firebase",
    # CS Fundamentals
    "algorithms":            "algorithms",
    "algoritms":             "algorithms",
    "data structure":        "data_structures",
    "data structures":       "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux":                 "ui_ux",
    "ui ux":                 "ui_ux",
    "figma":                 "figma",
}


RESUMES_RAW = [
    {"id": "01", "name": "Arjun Sharma",    "raw": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair",      "raw": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta",     "raw": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel",     "raw": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh",    "raw": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan", "raw": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta",     "raw": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao",     "raw": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar",    "raw": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer",      "raw": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

JDS_RAW = [
    {
        "id": "JD1", "company": "Kakao (Seoul)", "role": "ML Engineer",
        "raw": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    },
    {
        "id": "JD2", "company": "Naver (Seongnam)", "role": "Backend Engineer",
        "raw": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis",
    },
    {
        "id": "JD3", "company": "Line (Seoul)", "role": "Frontend Engineer",
        "raw": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS",
    },
]

TOTAL_RESUMES = 10   # corpus size fixed by problem statement




def normalize_skills(raw: str) -> list[str]:

    seen: set[str] = set()
    result: list[str] = []

    for token in raw.split(","):
        token = token.strip().lower()
        canonical = SKILL_ALIASES.get(token)
        if canonical is None:
            continue                          # discard unknown token
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result



def build_vocabulary(resumes: list[dict]) -> list[str]:
    """
    Sorted alphabetical union of all normalized, deduplicated resume skills.
    JD skills are NOT included.
    """
    vocab: set[str] = set()
    for r in resumes:
        vocab.update(r["skills"])
    return sorted(vocab)




def compute_tfidf(resumes: list[dict], vocab: list[str]) -> dict[str, list[float]]:
    """
    Returns {resume_id: tfidf_vector}
    Vector is aligned to `vocab` order.
    """
    # Step A: document frequency
    df: dict[str, int] = {skill: 0 for skill in vocab}
    for r in resumes:
        for skill in r["skills"]:          # already deduplicated
            df[skill] += 1

    # Step B: IDF per skill
    idf: dict[str, float] = {
        skill: math.log(TOTAL_RESUMES / df[skill])
        for skill in vocab
    }

    # Step C: TF-IDF vector per resume
    vocab_index = {skill: i for i, skill in enumerate(vocab)}
    vectors: dict[str, list[float]] = {}

    for r in resumes:
        skills_set = set(r["skills"])
        N = len(r["skills"])               # unique skill count
        tf = 1.0 / N                       # same TF for every skill (post-dedup)
        vec = [0.0] * len(vocab)
        for skill in r["skills"]:
            idx = vocab_index[skill]
            vec[idx] = tf * idf[skill]
        vectors[r["id"]] = vec

    return vectors, idf



def build_jd_vectors(jds_raw: list[dict], vocab: list[str]) -> list[dict]:
    """
    Normalize each JD's skill string, build a binary vector over vocab.
    Returns enriched JD list with 'normalized_skills' and 'vector' keys.
    """
    vocab_index = {skill: i for i, skill in enumerate(vocab)}
    jds = []

    for jd in jds_raw:
        norm_skills = normalize_skills(jd["raw"])          # reuse same normaliser
        vec = [0.0] * len(vocab)
        for skill in norm_skills:
            if skill in vocab_index:                        # must exist in vocabulary
                vec[vocab_index[skill]] = 1.0
        jds.append({**jd, "normalized_skills": norm_skills, "vector": vec})

    return jds




def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)



def rank_top3(
    jd: dict,
    resumes: list[dict],
    resume_vectors: dict[str, list[float]],
) -> list[dict]:
    scores = []
    for r in resumes:
        sim = cosine_similarity(resume_vectors[r["id"]], jd["vector"])
        scores.append({"name": r["name"], "score": sim})

    # Sort: descending score, then ascending name for ties
    scores.sort(key=lambda x: (-x["score"], x["name"]))
    return scores[:3]



W = 72

def sep(char="─"): print(char * W)



def print_normalization(resumes: list[dict]):
    sep()
    print("  STAGE 1-2 · NORMALIZATION + DEDUPLICATION")
    sep()
    print(f"  {'ID':<4} {'Name':<20} {'N':>3}  Canonical Skills")
    sep()
    for r in resumes:
        skills_str = ", ".join(r["skills"])
        print(f"  {r['id']:<4} {r['name']:<20} {len(r['skills']):>3}  {skills_str}")
    print()

def print_vocabulary(vocab: list[str]):
    sep()
    print(f"  STAGE 3 · VOCABULARY  ({len(vocab)} unique skills, alphabetically sorted)")
    sep()
    cols = 4
    for i in range(0, len(vocab), cols):
        row = vocab[i:i+cols]
        print("  " + "  ".join(f"{s:<25}" for s in row))
    print()

def print_tfidf(resumes: list[dict], resume_vectors: dict, vocab: list[str], idf: dict):
    sep()
    print("  STAGE 4 · TF-IDF VECTORS  (non-zero entries only)")
    sep()
    for r in resumes:
        vec = resume_vectors[r["id"]]
        N = len(r["skills"])
        tf = 1.0 / N
        print(f"\n  [{r['id']}] {r['name']}  (N={N}, TF={tf:.6f})")
        entries = [
            f"    {vocab[i]}: IDF={math.log(TOTAL_RESUMES/sum(1 for x in resumes if vocab[i] in x['skills'])):.6f}  "
            f"TF-IDF={vec[i]:.6f}"
            for i in range(len(vocab)) if vec[i] > 0
        ]
        for e in entries:
            print(e)
    print()

def print_jd_vectors(jds: list[dict], vocab: list[str]):
    sep()
    print("  STAGE 5 · JD BINARY VECTORS  (skills present in vocabulary)")
    sep()
    for jd in jds:
        present = [vocab[i] for i, v in enumerate(jd["vector"]) if v == 1.0]
        dropped = [s for s in jd["normalized_skills"] if s not in vocab]
        print(f"\n  {jd['id']} — {jd['company']} ({jd['role']})")
        print(f"    Matched  : {', '.join(present)}")
        if dropped:
            print(f"    Dropped  : {', '.join(dropped)}  (not in resume vocab)")
    print()

def print_similarity_matrix(resumes, jds, resume_vectors):
    sep()
    print("  STAGE 6 · COSINE SIMILARITY MATRIX")
    sep()
    header = f"  {'Candidate':<20}" + "".join(f"  {jd['id']:>10}" for jd in jds)
    print(header)
    sep()
    for r in resumes:
        row = f"  {r['name']:<20}"
        for jd in jds:
            sim = cosine_similarity(resume_vectors[r["id"]], jd["vector"])
            row += f"  {sim:>10.4f}"
        print(row)
    print()

def print_final_rankings(jds, resumes, resume_vectors):
    print("═" * W)
    print("  STAGE 7 · FINAL OUTPUT  —  Top 3 per JD")
    print("═" * W)
    for jd in jds:
        top3 = rank_top3(jd, resumes, resume_vectors)
        print(f"\n  {jd['id']} — {jd['company']} ({jd['role']})")
        result_str = ", ".join(f"{r['name']}({r['score']:.2f})" for r in top3)
        print(f"  {result_str}")
    print()
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  Orchestrate all stages
# ══════════════════════════════════════════════════════════════════════════════

def main():


    # ── Stage 1 & 2: Normalize + Deduplicate ─────────────────────────────────
    resumes = []
    for r in RESUMES_RAW:
        skills = normalize_skills(r["raw"])
        resumes.append({**r, "skills": skills})

    print_normalization(resumes)

    # ── Stage 3: Vocabulary ───────────────────────────────────────────────────
    vocab = build_vocabulary(resumes)
    print_vocabulary(vocab)

    # ── Stage 4: TF-IDF vectors (resumes only) ────────────────────────────────
    resume_vectors, idf = compute_tfidf(resumes, vocab)
    print_tfidf(resumes, resume_vectors, vocab, idf)

    # ── Stage 5: JD binary vectors ────────────────────────────────────────────
    jds = build_jd_vectors(JDS_RAW, vocab)
    print_jd_vectors(jds, vocab)

    # ── Stage 6: Cosine similarity matrix ─────────────────────────────────────
    print_similarity_matrix(resumes, jds, resume_vectors)

    # ── Stage 7: Rank + Final Output ──────────────────────────────────────────
    print_final_rankings(jds, resumes, resume_vectors)


if __name__ == "__main__":
    main()