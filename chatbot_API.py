"""
Core chatbot functionality extracted from ai_chatbot.py
"""
import os
import re
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import difflib
import google.generativeai as genai


# ===============================================================
# 🛑 NEWSENSE CLIENT
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
            resp = self.session.get(f"{self.base_url}/tenant/devices", headers=headers, params={"pageSize": 100, "page": page})
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
        """Check if data is available for a specific device and variable"""
        # Find device ID by name
        device_id = None
        for device in self.get_devices():
            if device['name'] == device_name:
                device_id = device['id']
                break
        
        if not device_id:
            return False, f"Device '{device_name}' not found"
        
        # Get available keys for this device
        available_keys = self.get_keys(device_id)
        
        if variable_name in available_keys:
            return True, f"Data available for device '{device_name}' and variable '{variable_name}'"
        else:
            return False, f"Variable '{variable_name}' not found for device '{device_name}'. Available variables: {', '.join(available_keys[:10])}"

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
                interval = 86400000 * 7
                agg = "AVG"
            elif duration_days > 30:
                interval = 86400000
                agg = "AVG"
            elif duration_days > 7:
                interval = 3600000
                agg = "AVG"
            else:
                interval = None
                agg = None

        except ValueError:
            return pd.DataFrame()

        params = {
            "startTs": start_ts,
            "endTs": end_ts,
            "keys": key,
        }

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
# 🕓 INTERPRET RELATIVE TIME
# ===============================================================
def interpret_relative_time(query: str):
    """
    Enhanced version to handle 'ngày', 'tuần', 'tháng', 'năm' and 'mới nhất/hiện tại'.
    Returns (start_date, end_date, is_latest_requested)
    """
    now = datetime.now()
    text = query.lower()
    start = None
    end = now
    
    # Detect if user is asking for the latest/current values
    is_latest = any(word in text for word in ["mới nhất", "hiện tại", "giá trị bao nhiêu", "lần cuối", "is what value"])

    if "hôm nay" in text:
        start = now.replace(hour=0, minute=0, second=0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest

    if "hôm qua" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), is_latest

    m = re.search(r"(\d+)\s*(ngày|tuần|tháng|năm)", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)

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

    # If asking for latest but no specific date context, look at last 24h
    if is_latest:
        start = now - timedelta(days=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), True

    return None, None, False



# ===============================================================
# 🧩 GEMINI PROMPT + chatbot()
# ===============================================================
SYSTEM_PROMPT = """
Trợ lý phân tích Newsense.
NV:
1) Xác định vị trí, loại dữ liệu, khoảng thời gian.
2) Chọn thiết bị từ KG.
3) Trả JSON, không giải thích:
{
  "location": "...",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "is_latest": false,
  "devices": [{"Device": "...", "Tên biến": "...", "Tên thiết bị": "..."}]
}
Lưu ý: "is_latest": true nếu người dùng hỏi giá trị hiện tại/mới nhất.
"""



def chatbot(query: str, kg_df: pd.DataFrame, chat_history: list, gemini_model):
    # Optimize KG representation once and reuse if possible (or just keep it lean here)
    compact_kg = kg_df[["Tên thiết bị", "Device", "Tên biến", "Vị trí", "Loại thiết bị"]].to_dict(orient="records")
    
    # Compress history significantly (last 3 turns only) for speed
    history_context = chat_history[-6:] if len(chat_history) > 6 else chat_history
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_context])
    
    # Pre-calculate time context to guide model and reduce its workload
    rel_start, rel_end, is_latest = interpret_relative_time(query)
    time_hint = f"Requested: {rel_start} to {rel_end}, Latest: {is_latest}" if rel_start else ""

    system_msg = f"{SYSTEM_PROMPT}\nKG:\n{json.dumps(compact_kg, ensure_ascii=False)}\nTimeHint: {time_hint}"
    full_prompt = f"{system_msg}\n\nhistory:\n{history_text}\n\nUser: {query}\nJSON:"

    # Aggressive generation config for speed
    gen_config = genai.HistoryConfig(
        # Use simple generation config
    )
    
    response = gemini_model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=800  # JSON for extraction is small
        )
    )
    
    content = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except:
        m = re.search(r"(\{.*\})", content, re.DOTALL)
        if m: result = json.loads(m.group(1))
        else: return None, chat_history

    # Meta enrichment
    if is_latest: result['is_latest'] = True
    
    if rel_start and rel_end:
        result['start_date'], result['end_date'] = rel_start, rel_end
    else:
        # Fallback normalization logic
        def norm(s):
            if not s: return None
            for f in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try: return datetime.strptime(s, f).strftime("%Y-%m-%d")
                except: continue
            return None
        result['start_date'], result['end_date'] = norm(result.get('start_date')), norm(result.get('end_date'))
        
        now = datetime.now()
        if not result.get('end_date'): result['end_date'] = now.strftime("%Y-%m-%d")
        if not result.get('start_date'):
            days = 1 if result.get('is_latest') else 30
            result['start_date'] = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
    return result, chat_history



def analyze_data(fetched_data_list, original_query, gemini_model):
    """Gửi TẤT CẢ dữ liệu tóm tắt đến Gemini trong MỘT lần gọi duy nhất."""
    if not fetched_data_list:
        return "Không có dữ liệu để phân tích."

    stats_list = []
    for item in fetched_data_list:
        df = item['data']
        label = item['label']

        if df.empty or 'value' not in df.columns:
            continue

        stats = {
            "ten_bien": label,
            "so_luong_diem_du_lieu": len(df),
            "gia_tri_trung_binh": round(df['value'].mean(), 2),
            "gia_tri_thap_nhat": round(df['value'].min(), 2),
            "gia_tri_cao_nhat": round(df['value'].max(), 2),
            "ngay_bat_dau_du_lieu": df['ts'].min().strftime("%Y-%m-%d"),
            "ngay_ket_thuc_du_lieu": df['ts'].max().strftime("%Y-%m-%d")
        }
        stats_list.append(stats)

    if not stats_list:
        return "Không có dữ liệu hợp lệ nào được tìm thấy để phân tích."

    analysis_prompt = f"""
    Bạn là một kỹ sư phân tích dữ liệu. Người dùng vừa xem biểu đồ cho câu hỏi: "{original_query}"

    Dưới đây là một DANH SÁCH các số liệu thống kê tóm tắt cho các biến:
    {json.dumps(stats_list, ensure_ascii=False, indent=2)}

    Hãy đưa ra phân tích chuyên môn 2-3 câu bằng tiếng Việt cho TỪNG BIẾN trong danh sách.
    Tập trung vào: giá trị trung bình, xu hướng, và các điểm bất thường (giá trị cao nhất/thấp nhất so với trung bình).
    Viết như một kỹ sư: kỹ thuật, ngắn gọn, và có thể hành động.

    Định dạng phản hồi của bạn:
    --- Phân tích cho: [Tên biến] ---
    [2-3 câu phân tích kỹ thuật bằng tiếng Việt]

    --- Phân tích cho: [Tên biến] ---
    [2-3 câu phân tích kỹ thuật bằng tiếng Việt]
    """

    try:
        config = genai.GenerationConfig(max_output_tokens=8192)
        response = gemini_model.generate_content(analysis_prompt, generation_config=config)

        if not response.parts:
            return "Lỗi phân tích: Model không trả về nội dung."
        else:
            return response.text.strip()

    except Exception as e:
        return f"Lỗi khi tạo phân tích tổng hợp: {e}"

