"""
Core chatbot functionality - OPTIMIZED with Embedding + Cosine Similarity
==========================================================================
Thay đổi so với bản gốc:
1. EmbeddingKGMatcher: Embed KG 1 lần, mỗi query chỉ embed 1 câu → cosine similarity (0 token LLM)
2. try_local_parse(): Parse query đơn giản hoàn toàn local (0 token)
3. chatbot(): Chỉ gửi top-K candidates cho Gemini thay vì toàn bộ KG (~95% token saved)
4. analyze_data(): Prompt compact hơn, max_output_tokens động (~50% token saved)
5. compact_history(): Chỉ giữ 2 câu hỏi gần nhất của user (70% token history saved)

Ước tính tiết kiệm tổng: 70-95% token / query
"""
import os
import re
import json
import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import difflib
import google.generativeai as genai


# ===============================================================
# 🛑 NEWSENSE CLIENT (giữ nguyên)
# ===============================================================
class NewsenseClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.token = self.login()

    def login(self):
        url = f"{self.base_url}/auth/login"
        resp = self.session.post(url, json={"username": self.username, "password": self.password})
        if resp.status_code == 401:
            raise Exception("❌ Sai username hoặc mật khẩu.")
        resp.raise_for_status()
        return resp.json().get("token")

    def get_devices(self):
        headers = {"X-Authorization": f"Bearer {self.token}"}
        page = 0
        devices = []
        while True:
            resp = self.session.get(
                f"{self.base_url}/tenant/devices",
                headers=headers, params={"pageSize": 100, "page": page}
            )
            resp.raise_for_status()
            data = resp.json()
            for d in data.get("data", []):
                devices.append({"id": d["id"]["id"], "name": d["name"]})
            if not data.get("hasNextPage"):
                break
            page += 1
        return devices

    def get_keys(self, device_id):
        headers = {"X-Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/plugins/telemetry/DEVICE/{device_id}/keys/timeseries"
        resp = self.session.get(url, headers=headers)
        if resp.status_code == 200:
            return [k for k in resp.json() if k != 'timestamp']
        else:
            return []

    def check_data_availability(self, device_name, variable_name):
        device_id = None
        for device in self.get_devices():
            if device['name'] == device_name:
                device_id = device['id']
                break
        if not device_id:
            return False, f"Device '{device_name}' not found"
        available_keys = self.get_keys(device_id)
        if variable_name in available_keys:
            return True, f"Data available for device '{device_name}' and variable '{variable_name}'"
        else:
            return False, f"Variable '{variable_name}' not found for device '{device_name}'. Available: {', '.join(available_keys[:10])}"

    def get_timeseries(self, device_id, key, start_date_str, end_date_str):
        headers = {"X-Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            duration_days = (end_dt - start_dt).days
            if duration_days > 90:
                interval, agg = 86400000 * 7, "AVG"
            elif duration_days > 30:
                interval, agg = 86400000, "AVG"
            elif duration_days > 7:
                interval, agg = 3600000, "AVG"
            else:
                interval, agg = None, None
        except ValueError:
            return pd.DataFrame()

        params = {"startTs": start_ts, "endTs": end_ts, "keys": key}
        if agg and interval:
            params["interval"] = interval
            params["agg"] = agg
        else:
            params["limit"] = 10000

        resp = self.session.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json().get(key, [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=['value'])
        return df[["ts", "value"]]


# ===============================================================
# 🕓 INTERPRET RELATIVE TIME (giữ nguyên)
# ===============================================================
def interpret_relative_time(query: str):
    now = datetime.now()
    text = query.lower()
    start = None
    end = now
    is_latest = any(word in text for word in [
        "mới nhất", "hiện tại", "giá trị bao nhiêu", "lần cuối", "is what value"
    ])

    if "hôm nay" in text:
        start = now.replace(hour=0, minute=0, second=0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest
    if "hôm qua" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest

    m = re.search(r"(\d+)\s*(ngày|tuần|tháng|năm)", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "ngày":
            start = now - timedelta(days=n)
        elif unit == "tuần":
            start = now - timedelta(weeks=n)
        elif unit == "tháng":
            start = now - timedelta(days=n * 30)
        elif unit == "năm":
            start = datetime(now.year - n + 1, 1, 1)
        if start:
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest

    if "tuần này" in text:
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest
    if "tháng này" in text:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest
    if "từ đầu năm" in text:
        start = now.replace(day=1, month=1, hour=0, minute=0, second=0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest
    if "năm ngoái" in text:
        y = now.year - 1
        return f"{y}-01-01", f"{y}-12-31", is_latest
    if is_latest:
        start = now - timedelta(days=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), True

    return None, None, False


# ===============================================================
# ⭐ MỚI: EMBEDDING KG MATCHER
# Embed KG 1 lần → mỗi query chỉ embed 1 câu + cosine (0 token LLM)
# ===============================================================
class EmbeddingKGMatcher:
    """
    Thay thế việc gửi toàn bộ KG cho Gemini.

    Init (1 lần): embed toàn bộ KG → cache vectors
    Mỗi query:   embed 1 câu (~20 token embedding, rất rẻ)
                  → cosine similarity (0 token, numpy local)
                  → trả top-K candidates

    So sánh token (KG 200 dòng):
    ┌───────────────────┬────────────┬────────────┐
    │                   │ Bản gốc    │ Embedding  │
    ├───────────────────┼────────────┼────────────┤
    │ KG input tokens   │ ~4,000     │ 0          │
    │ Embedding tokens  │ 0          │ ~20 (rẻ)   │
    │ Gemini confirm    │ n/a        │ ~80-150    │
    │ TỔNG / query      │ ~4,500     │ ~100-200   │
    └───────────────────┴────────────┴────────────┘
    """

    CACHE_FILE = "kg_embeddings_cache.pkl"

    def __init__(self, kg_df: pd.DataFrame, model_name: str = "models/gemini-embedding-001"):
        self.kg_df = kg_df.copy()
        self.model_name = model_name
        self.kg_embeddings = None  # np.ndarray (N, dim)
        self.kg_texts = []         # list[str] mô tả từng device

        self._build_or_load_index()

    def _build_or_load_index(self):
        """Load cache nếu có, không thì embed rồi save cache."""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
                # Kiểm tra cache còn khớp KG hiện tại không
                if cache.get("kg_hash") == self._kg_hash():
                    self.kg_texts = cache["texts"]
                    self.kg_embeddings = cache["embeddings"]
                    print(f"✅ Loaded embedding cache ({len(self.kg_texts)} devices)")
                    return
            except Exception:
                pass  # Cache hỏng → rebuild

        self._build_index()
        self._save_cache()

    def _kg_hash(self) -> str:
        """Hash đơn giản để detect KG thay đổi."""
        raw = self.kg_df.to_json(force_ascii=False)
        return str(hash(raw))

    def _build_index(self):
        """Embed toàn bộ KG — chỉ chạy 1 lần."""
        self.kg_texts = []
        for _, row in self.kg_df.iterrows():
            text = (
                f"{row.get('Tên thiết bị', '')} đo {row.get('Tên biến', '')} "
                f"tại {row.get('Vị trí lắp', '')} loại {row.get('Loại thiết bị', '')}"
            )
            self.kg_texts.append(text)

        result = genai.embed_content(
            model=self.model_name,
            content=self.kg_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        self.kg_embeddings = np.array(result['embedding'])
        print(f"✅ Built embedding index: {self.kg_embeddings.shape}")

    def _save_cache(self):
        """Lưu cache để lần sau không cần embed lại."""
        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump({
                    "kg_hash": self._kg_hash(),
                    "texts": self.kg_texts,
                    "embeddings": self.kg_embeddings,
                }, f)
            print(f"✅ Saved embedding cache to {self.CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Tìm top-K devices phù hợp nhất.
        Chi phí: ~20 token embedding + 0 token LLM
        """
        query_result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_vec = np.array(query_result['embedding'])

        # Cosine similarity — hoàn toàn local
        similarities = self._cosine_similarity(query_vec, self.kg_embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.kg_df.iloc[idx]
            results.append({
                "Device": row.get("Device", ""),
                "Tên biến": row.get("Tên biến", ""),
                "Tên thiết bị": row.get("Tên thiết bị", ""),
                "Vị trí lắp": row.get("Vị trí lắp", ""),
                "Loại thiết bị": row.get("Loại thiết bị", ""),
                "similarity": float(similarities[idx]),
            })
        return results

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)


# ===============================================================
# ⭐ MỚI: LOCAL PARSE — 0 TOKEN KHI QUERY ĐƠN GIẢN
# ===============================================================
def try_local_parse(query: str, kg_matcher: EmbeddingKGMatcher):
    """
    Thử trả kết quả hoàn toàn local bằng embedding search + time parse.
    Nếu similarity cao (>0.75) và parse được thời gian → KHÔNG cần gọi Gemini.

    Token tiết kiệm: 100% LLM tokens cho những query rõ ràng.
    """
    rel_start, rel_end, is_latest = interpret_relative_time(query)
    if not rel_start:
        return None  # Không parse được thời gian → cần Gemini

    candidates = kg_matcher.search(query, top_k=3)
    if not candidates or candidates[0]['similarity'] < 0.75:
        return None  # Không đủ confident → cần Gemini

    # Confident match → trả kết quả local
    devices = [
        {"Device": c["Device"], "Tên biến": c["Tên biến"], "Tên thiết bị": c["Tên thiết bị"]}
        for c in candidates if c['similarity'] > 0.5
    ]
    if not devices:
        return None

    return {
        "location": candidates[0].get("Vị trí lắp", ""),
        "start_date": rel_start,
        "end_date": rel_end,
        "is_latest": is_latest,
        "devices": devices,
    }


# ===============================================================
# ⭐ MỚI: COMPACT HISTORY — giảm 70% token history
# ===============================================================
def compact_history(chat_history: list, max_turns: int = 2) -> str:
    """Chỉ giữ vài câu hỏi gần nhất của user, bỏ JSON response dài."""
    user_msgs = [m['content'] for m in chat_history if m['role'] == 'user']
    recent = user_msgs[-max_turns:]
    if not recent:
        return ""
    return "Prev:" + "|".join(recent)


# ===============================================================
# 🧩 CHATBOT — OPTIMIZED
# ===============================================================
SYSTEM_PROMPT_COMPACT = """Chọn thiết bị phù hợp từ danh sách candidates cho câu hỏi.
Trả JSON duy nhất:
{"location":"...","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD","is_latest":false,"devices":[{"Device":"...","Tên biến":"...","Tên thiết bị":"..."}]}
is_latest=true nếu hỏi giá trị hiện tại/mới nhất."""


def chatbot(query: str, kg_df: pd.DataFrame, chat_history: list, gemini_model,
            kg_matcher: EmbeddingKGMatcher = None):
    """
    Chatbot tối ưu token:

    Luồng xử lý:
    1. try_local_parse() — nếu query rõ ràng → 0 token LLM ✅
    2. Embedding search top-K → chỉ gửi K candidates cho Gemini (thay vì toàn bộ KG)
    3. Prompt compact + history compact

    Yêu cầu: truyền thêm kg_matcher (khởi tạo 1 lần ở app startup)
    Nếu không có kg_matcher → fallback về cách cũ (gửi toàn bộ KG)
    """

    # ── Bước 1: Thử parse local (0 token LLM) ──
    if kg_matcher:
        local_result = try_local_parse(query, kg_matcher)
        if local_result:
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": json.dumps(local_result, ensure_ascii=False)})
            return local_result, chat_history

    # ── Bước 2: Cần Gemini — nhưng gửi ít nhất có thể ──
    rel_start, rel_end, is_latest = interpret_relative_time(query)
    time_hint = f"T:{rel_start}~{rel_end},L:{is_latest}" if rel_start else ""
    hist_str = compact_history(chat_history, max_turns=2)

    if kg_matcher:
        # ⭐ CHỈ GỬI TOP-K CANDIDATES thay vì toàn bộ KG
        candidates = kg_matcher.search(query, top_k=10)
        candidates_str = "\n".join(
            f"{c['Tên thiết bị']}|{c['Device']}|{c['Tên biến']}|{c['Vị trí lắp']}|{c['Loại thiết bị']}"
            for c in candidates
        )
        full_prompt = (
            f"{SYSTEM_PROMPT_COMPACT}\n"
            f"Candidates:\n{candidates_str}\n"
            f"{time_hint}\n{hist_str}\n"
            f"Q:{query}\nJSON:"
        )
    else:
        # Fallback: gửi toàn bộ KG (bản gốc)
        # Dùng các cột có sẵn trong kg_df, tránh KeyError
        desired_cols = ["Tên thiết bị", "Device", "Tên biến", "Vị trí lắp", "Loại thiết bị"]
        available_cols = [c for c in desired_cols if c in kg_df.columns]
        if not available_cols:
            available_cols = list(kg_df.columns)  # Dùng tất cả cột nếu không match
        compact_kg = kg_df[available_cols].to_dict(orient="records")
        full_prompt = (
            f"{SYSTEM_PROMPT_COMPACT}\n"
            f"KG:\n{json.dumps(compact_kg, ensure_ascii=False)}\n"
            f"{time_hint}\n{hist_str}\n"
            f"Q:{query}\nJSON:"
        )

    response = gemini_model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=500,
        ),
    )

    content = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except Exception:
        m = re.search(r"(\{.*\})", content, re.DOTALL)
        if m:
            result = json.loads(m.group(1))
        else:
            return None, chat_history

    # ── Enrich với local time parsing ──
    if is_latest:
        result['is_latest'] = True
    if rel_start and rel_end:
        result['start_date'], result['end_date'] = rel_start, rel_end
    else:
        def norm(s):
            if not s:
                return None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except Exception:
                    continue
            return None

        result['start_date'] = norm(result.get('start_date'))
        result['end_date'] = norm(result.get('end_date'))
        now = datetime.now()
        if not result.get('end_date'):
            result['end_date'] = now.strftime("%Y-%m-%d")
        if not result.get('start_date'):
            days = 1 if result.get('is_latest') else 30
            result['start_date'] = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
    return result, chat_history


# ===============================================================
# 📊 ANALYZE DATA — OPTIMIZED
# ===============================================================
def analyze_data(fetched_data_list, original_query, gemini_model):
    """
    Tối ưu so với bản gốc:
    - Stats dạng compact 1 dòng/biến (thay vì JSON object dài)
    - Prompt ngắn hơn ~50%
    - max_output_tokens tính động theo số biến
    """
    if not fetched_data_list:
        return "Không có dữ liệu để phân tích."

    stats_lines = []
    for item in fetched_data_list:
        df = item['data']
        label = item['label']
        if df.empty or 'value' not in df.columns:
            continue
        line = (
            f"{label}: n={len(df)}, "
            f"avg={df['value'].mean():.2f}, "
            f"min={df['value'].min():.2f}, "
            f"max={df['value'].max():.2f}, "
            f"{df['ts'].min().strftime('%Y-%m-%d')}~{df['ts'].max().strftime('%Y-%m-%d')}"
        )
        stats_lines.append(line)

    if not stats_lines:
        return "Không có dữ liệu hợp lệ nào được tìm thấy để phân tích."

    stats_text = "\n".join(stats_lines)

    analysis_prompt = (
        f"Phân tích kỹ thuật ngắn gọn (2-3 câu/biến, tiếng Việt) cho:\n"
        f"Q: \"{original_query}\"\n"
        f"Data:\n{stats_text}\n"
        f"Focus: avg, xu hướng, bất thường. Viết như kỹ sư, ngắn gọn."
    )

    try:
        max_tokens = min(len(stats_lines) * 150 + 100, 4096)
        response = gemini_model.generate_content(
            analysis_prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )
        if not response.parts:
            return "Lỗi phân tích: Model không trả về nội dung."
        return response.text.strip()
    except Exception as e:
        return f"Lỗi khi tạo phân tích tổng hợp: {e}"


# ===============================================================
# 🚀 HƯỚNG DẪN TÍCH HỢP
# ===============================================================
"""
CÁCH SỬ DỤNG (thay đổi tối thiểu ở app chính):

1. KHỞI TẠO (1 lần khi start app) — thêm 2 dòng:

    from chatbot_core import EmbeddingKGMatcher
    
    kg_df = pd.read_excel("knowledge_graph.xlsx")
    kg_matcher = EmbeddingKGMatcher(kg_df)         # ← THÊM DÒNG NÀY
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

2. GỌI CHATBOT — thêm tham số kg_matcher:

    # Trước:
    result, history = chatbot(query, kg_df, chat_history, gemini_model)
    
    # Sau:
    result, history = chatbot(query, kg_df, chat_history, gemini_model, kg_matcher=kg_matcher)
    #                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^

3. analyze_data() — KHÔNG CẦN THAY ĐỔI GÌ, đã tự tối ưu bên trong.

4. KHI KG THAY ĐỔI (thêm/sửa device):
    
    kg_df = pd.read_excel("knowledge_graph_updated.xlsx")
    kg_matcher = EmbeddingKGMatcher(kg_df)  # Tự rebuild + cache mới

Vậy thôi! Chỉ cần thêm 1 dòng init + 1 tham số là xong.
"""