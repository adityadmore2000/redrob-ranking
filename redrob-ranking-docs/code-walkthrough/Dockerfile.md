# `Dockerfile` — Deep Line-by-Line Teaching Notes

> **Packages the Streamlit demo into a container for Hugging Face (HF) Spaces.** A
> Dockerfile is a recipe: each instruction builds one **layer** of an image that can be
> run identically anywhere. This one is small but hits several best practices worth knowing
> for an interview: non-root user, layer-cache ordering, and env-based configuration.

---

## Line 1–2 — Base image

```dockerfile
# Hugging Face Spaces — Docker SDK deployment for the Streamlit demo.
FROM python:3.10-slim
```

- **`FROM`** sets the **base image** every build starts from. `python:3.10-slim` is the official
  Python 3.10 image on a **slim** (stripped-down Debian) base — much smaller than the full image
  (fewer OS packages), which means faster pulls and a smaller attack surface, while still having
  `pip` and the Python runtime.
- **Interview:** why `-slim` (size/security) and the tradeoff (you may need to `apt-get install`
  build tools if a dependency needs compiling).

---

## Line 4–5 — Non-root user

```dockerfile
# HF Spaces requires the container to run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
```

- **`RUN`** executes a shell command **at build time**, baking the result into the image.
- Creates a user named `user` with **uid 1000** and a home directory (`-m`). HF Spaces mandates
  running as this non-root uid.
- **Security best practice generally:** don't run containers as root; if the app is compromised, a
  non-root user limits the blast radius.
- **Interview:** why run as non-root; what `RUN` bakes into the image.

---

## Line 9–13 — Environment configuration

```dockerfile
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

- **`ENV`** sets environment variables for both later build steps and the running container. One
  `ENV` with `\` line-continuations sets several at once (**fewer layers** than multiple `ENV`s).
- `HOME=/home/user` — the non-root user's home.
- `PATH=/home/user/.local/bin:$PATH` — **prepends** the location where `pip install --user` puts
  console scripts (like `streamlit`), so they're runnable by name.
- `HF_HOME=/home/user/.cache/huggingface` — tells Hugging Face libraries **where to cache the
  downloaded model** (BAAI/bge-base-en-v1.5). Pointing it into the user's home ensures a
  **writable, predictable** location (the container filesystem can be read-only elsewhere).
- `STREAMLIT_SERVER_PORT=7860` — HF Spaces expects apps on **port 7860**.
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0` — bind to **all interfaces** (not just localhost) so the
  container is reachable from outside. Streamlit reads these `STREAMLIT_*` env vars as config.
- **Interview:** why one `ENV` (layer count); why `0.0.0.0` vs `127.0.0.1` in containers; env-based
  config for the HF cache and port.

---

## Line 15 — Working directory

```dockerfile
WORKDIR /home/user/app
```

- **`WORKDIR`** sets the current directory for subsequent instructions (and the running container),
  creating it if needed. All later `COPY`/`RUN`/`CMD` are relative to here.

---

## Line 17–20 — Dependencies first (the layer-cache trick)

```dockerfile
# Install dependencies first so this layer caches across app-code changes.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
```

- **The single most important optimization here.** Docker caches each layer and **reuses it if the
  layer's inputs haven't changed.** By copying **only `requirements.txt`** and installing *before*
  copying the app code, the expensive `pip install` layer is **cached** and only re-runs when
  dependencies change — not on every code edit. If you copied everything first, any code change
  would bust the cache and force a full reinstall.
- `COPY --chown=user` — copy the file and set ownership to the non-root `user` in one step.
- `--no-cache-dir` — tells pip **not** to keep its download cache, shrinking the image (the cache is
  useless inside a built image).
- `--upgrade pip` first ensures a modern resolver.
- `&&` chains both installs in **one `RUN`** = one layer.
- **Interview gold:** explain Docker layer caching and *why* `requirements.txt` is copied before the
  app code. This is the classic Dockerfile question.

---

## Line 22–23 — Copy the application

```dockerfile
# Copy the application (Dockerfile context is pruned via .dockerignore).
COPY --chown=user . .
```

- Copies the rest of the project into `WORKDIR`, owned by `user`. This layer *does* rebust on code
  changes — but that's cheap; the heavy deps layer above stays cached.
- The comment notes a **`.dockerignore`** prunes the build context (excludes `.venv`, `.git`, data,
  etc.) so they aren't sent to the daemon or copied in — keeping the image small and builds fast.
- **Interview:** what `.dockerignore` does (analogous to `.gitignore` for the build context).

---

## Line 26 — Drop privileges

```dockerfile
USER user
```

- **`USER`** switches the user for all subsequent instructions **and the running container**. Build
  steps that needed root (creating the user) ran earlier; from here on everything runs as the
  unprivileged `user`. This is the "do setup as root, then drop privileges" pattern.

---

## Line 28 — Document the port

```dockerfile
EXPOSE 7860
```

- **`EXPOSE`** is **documentation/metadata** — it declares the port the app listens on but does
  **not** actually publish it (that's done at `docker run -p` time or by the platform). HF Spaces
  reads this to route traffic to 7860.
- **Interview:** `EXPOSE` documents intent; it doesn't open the port by itself.

---

## Line 30 — Startup command

```dockerfile
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
```

- **`CMD`** defines the **default command run when the container starts** (contrast `RUN`, which
  runs at *build* time). This launches the Streamlit app.
- **Exec form** (`["a","b",...]`, a JSON array) — runs the binary directly **without a shell**, so
  signals (like `SIGTERM` on shutdown) reach the process correctly. Preferred over the shell form
  `CMD streamlit run ...`.
- Flags: `--server.port`/`--server.address` mirror the env vars (belt-and-suspenders);
  `--server.enableCORS=false` and `--server.enableXsrfProtection=false` relax browser protections —
  necessary because the app runs **behind HF Spaces' reverse proxy/iframe**, where the default CORS/
  XSRF checks would otherwise block it.
- **Interview:** `CMD` vs `RUN` (runtime vs build); exec form vs shell form and why exec form
  handles signals better; why CORS/XSRF are disabled behind a proxy.

---

## Big-picture takeaways

1. **Layer-cache ordering** — copy `requirements.txt` and install deps *before* copying app code, so
   the expensive install layer is reused across code changes. The headline best practice.
2. **Non-root by design** — create `user` (uid 1000), do privileged setup, then `USER user`.
3. **Env-driven config** — `HF_HOME` (writable model cache), port 7860, bind `0.0.0.0` — all wired
   for HF Spaces.
4. **Small image** — `-slim` base, `--no-cache-dir` pip, `.dockerignore`-pruned context.
5. **Correct startup** — exec-form `CMD` for signal handling; CORS/XSRF disabled to work behind the
   Spaces proxy.
