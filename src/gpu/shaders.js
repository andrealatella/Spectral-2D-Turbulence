const CACHE = new Map();

async function loadRaw(name) {
  if (!CACHE.has(name)) {
    const url = new URL(`../shaders/${name}.wgsl`, import.meta.url);
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`failed to load shader ${name}.wgsl (${res.status})`);
    }
    CACHE.set(name, await res.text());
  }
  return CACHE.get(name);
}

export function flt(x) {
  const s = String(x);
  return s.includes('.') || s.includes('e') || s.includes('E') ? s : `${s}.0`;
}

export function gridConstants(n) {
  return {
    N: n,
    NF: flt(n),
    NCELLS: n * n,
    KCUT: flt(n / 3),
    FORCE_NORM: flt(0.01 * n * n),
  };
}

export function fill(src, vars) {
  return src.replace(/\$([A-Z0-9_]+)\$/g, (_, key) => {
    if (!(key in vars)) {
      throw new Error(`unbound shader constant ${key}`);
    }
    return String(vars[key]);
  });
}

export async function shaderSource(name, vars, { prelude = true } = {}) {
  const body = await loadRaw(name);
  const head = prelude ? `${await loadRaw('common')}\n` : '';
  return fill(head + body, vars);
}

export async function shaderModule(device, name, vars, opts) {
  const code = await shaderSource(name, vars, opts);
  const module = device.createShaderModule({ label: name, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const detail = errors.map((m) => `  ${name}.wgsl:${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`WGSL compilation failed:\n${detail}`);
  }
  return module;
}
