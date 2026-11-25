const LANG_LABEL = { zh: "中文", ja: "日语", en: "英语" };
let VOICES = [];
let currentVoice = null;
let currentLang = null;
let busy = false;

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function init() {
  try {
    VOICES = await getJSON("/api/voices");
  } catch (e) {
    $("gen-hint").textContent = "无法连接后端：" + e.message;
    return;
  }
  renderVoices();
  if (VOICES.length) selectVoice(VOICES[0].id);
  loadHealth();
  loadGallery();
  $("speed").addEventListener("input", (e) => {
    $("speed-val").textContent = Number(e.target.value).toFixed(2) + "×";
  });
  $("text").addEventListener("input", () => {
    $("count").textContent = $("text").value.length;
  });
  $("generate").addEventListener("click", generate);
}

async function loadHealth() {
  try {
    const h = await getJSON("/api/health");
    setDot("gsv", h.gpt_sovits);
    setDot("cosy", h.cosy_worker);
  } catch (e) { /* ignore */ }
}

function setDot(key, ok) {
  const dot = $("dot-" + key);
  const st = $("st-" + key);
  dot.classList.toggle("ok", !!ok);
  dot.classList.toggle("bad", !ok);
  st.textContent = (key === "gsv" ? "GPT-SoVITS" : "CosyVoice") + (ok ? " 就绪" : " 未就绪");
}

function renderVoices() {
  const box = $("voices");
  box.innerHTML = "";
  VOICES.forEach((v) => {
    const el = document.createElement("div");
    el.className = "voice";
    el.dataset.id = v.id;
    el.innerHTML = `
      <span class="badge ${v.available ? "ok" : "bad"}">${v.available ? "可用" : "未就绪"}</span>
      <div class="vhead"><span class="vname">${v.name}</span><span class="vtag">${v.subtitle}</span></div>
      <div class="vdesc">${v.description}</div>
      <div class="vref"><span>参考音色</span><audio controls preload="none" src="${v.ref_audio}"></audio></div>
    `;
    el.addEventListener("click", () => selectVoice(v.id));
    box.appendChild(el);
  });
}

function selectVoice(id) {
  currentVoice = VOICES.find((v) => v.id === id);
  if (!currentVoice) return;
  document.querySelectorAll(".voice").forEach((el) =>
    el.classList.toggle("active", el.dataset.id === id));
  currentLang = currentVoice.default_lang;
  renderLangs();
  renderExamples();
  $("result").classList.add("hidden");
  $("gen-hint").textContent = "";
  $("gen-hint").classList.remove("err");
  updateGenButton();
}

function renderLangs() {
  const row = $("lang-row");
  row.innerHTML = "";
  currentVoice.langs.forEach((l) => {
    const b = document.createElement("button");
    b.className = "lang" + (l === currentLang ? " active" : "");
    b.textContent = LANG_LABEL[l] || l;
    b.addEventListener("click", () => {
      currentLang = l;
      renderLangs();
    });
    row.appendChild(b);
  });
}

function renderExamples() {
  const box = $("examples");
  box.innerHTML = "";
  (currentVoice.examples || []).forEach((t) => {
    const b = document.createElement("button");
    b.className = "ex";
    b.textContent = t;
    b.title = t;
    b.addEventListener("click", () => {
      $("text").value = t;
      $("count").textContent = t.length;
    });
    box.appendChild(b);
  });
}

function updateGenButton() {
  const btn = $("generate");
  const ok = currentVoice && currentVoice.available;
  btn.disabled = !ok || busy;
  if (!ok) {
    $("gen-hint").textContent = "该音色引擎未就绪：" + (currentVoice.error || "请检查后端");
    $("gen-hint").classList.add("err");
  } else if ($("gen-hint").classList.contains("err")) {
    $("gen-hint").textContent = "";
    $("gen-hint").classList.remove("err");
  }
}

async function generate() {
  if (busy || !currentVoice) return;
  const text = $("text").value.trim();
  if (!text) { $("gen-hint").textContent = "请先输入文本"; $("gen-hint").classList.add("err"); return; }
  busy = true;
  const btn = $("generate");
  const old = btn.textContent;
  btn.textContent = "生成中…";
  $("gen-hint").textContent = "正在合成，请稍候（首次加载较慢）…";
  $("gen-hint").classList.remove("err");
  updateGenButton();
  btn.disabled = true;

  try {
    const resp = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice: currentVoice.id,
        text,
        lang: currentLang,
        speed: Number($("speed").value),
      }),
    });
    if (!resp.ok) throw new Error((await resp.text()) || resp.statusText);
    const gen = resp.headers.get("X-Gen-Seconds");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const player = $("player");
    player.src = url;
    $("download").href = url;
    $("result").classList.remove("hidden");
    $("result-meta").textContent = gen ? `合成耗时 ${gen}s` : "";
    $("gen-hint").textContent = "生成完成 ✓";
    player.play().catch(() => {});
  } catch (e) {
    $("gen-hint").textContent = "生成失败：" + e.message;
    $("gen-hint").classList.add("err");
  } finally {
    busy = false;
    btn.textContent = old;
    updateGenButton();
  }
}

async function loadGallery() {
  let items = [];
  try { items = await getJSON("/api/samples"); } catch (e) { return; }
  const box = $("gallery");
  box.innerHTML = "";
  const groups = {};
  items.forEach((it) => { (groups[it.voice] = groups[it.voice] || []).push(it); });
  Object.keys(groups).forEach((vid) => {
    const g = document.createElement("div");
    g.className = "gal-group";
    const name = groups[vid][0].voice_name;
    g.innerHTML = `<h3>${name}</h3>`;
    groups[vid].forEach((it) => {
      const row = document.createElement("div");
      row.className = "gal-item";
      row.innerHTML = `
        <div class="info">
          <div class="gtitle">${it.title}<span class="gtag">${it.engine}</span></div>
          <div class="gtext">${it.text || ""}</div>
        </div>
        <audio controls preload="none" src="/samples/${encodeURIComponent(it.file)}"></audio>
      `;
      g.appendChild(row);
    });
    box.appendChild(g);
  });
}

init();
