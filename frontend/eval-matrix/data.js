const META_KEYS = new Set(["step", "timestamp"]);

export function numericMetrics(runLogs) {
  const metrics = new Set();
  for (const entry of runLogs) {
    for (const log of entry.logs || []) {
      for (const [key, value] of Object.entries(log)) {
        if (!META_KEYS.has(key) && typeof value === "number" && Number.isFinite(value)) {
          metrics.add(key);
        }
      }
    }
  }
  return [...metrics].sort((left, right) => {
    const leftEval = left.toLowerCase().includes("eval");
    const rightEval = right.toLowerCase().includes("eval");
    if (leftEval !== rightEval) return leftEval ? -1 : 1;
    return left.localeCompare(right, undefined, { numeric: true });
  });
}

export function defaultMetric(metrics) {
  return (
    metrics.find((metric) => /(^|\/)eval(?:\/|$).*\/(?:acc|accuracy)$/i.test(metric)) ||
    metrics.find((metric) => /eval.*(?:acc|accuracy)/i.test(metric)) ||
    metrics.find((metric) => /eval/i.test(metric)) ||
    metrics[0] ||
    ""
  );
}

export function buildMatrix(runLogs, runs, metric) {
  const logsByKey = new Map();
  for (const entry of runLogs) {
    logsByKey.set(entry.run_id || entry.run, entry.logs || []);
  }

  const steps = new Set();
  const rows = runs.map((run) => {
    const points = new Map();
    const logs = logsByKey.get(run.id || run.name) || [];
    for (const log of logs) {
      const step = Number(log.step);
      const value = log[metric];
      if (!Number.isFinite(step) || typeof value !== "number" || !Number.isFinite(value)) continue;
      points.set(step, { value, timestamp: log.timestamp || "" });
      steps.add(step);
    }
    const ordered = [...points.entries()].sort(([left], [right]) => left - right);
    const first = ordered.at(0);
    const latest = ordered.at(-1);
    return {
      id: run.id || run.name,
      name: run.name || "Unnamed run",
      createdAt: run.created_at || "",
      points,
      firstValue: first?.[1].value ?? null,
      latestStep: latest?.[0] ?? null,
      latestValue: latest?.[1].value ?? null,
      delta: first && latest ? latest[1].value - first[1].value : null,
    };
  });

  return { rows, steps: [...steps].sort((left, right) => left - right) };
}

export function parseStepFilter(expression) {
  const text = String(expression ?? "").trim();
  const all = { valid: true, match: () => true };
  if (!text) return all;

  const parseBound = (token) => {
    const value = token.trim();
    if (!value) return null;
    return /^[-+]?\d+(?:\.\d+)?$/.test(value) ? Number(value) : undefined;
  };
  const predicates = [];
  for (const rawTerm of text.split(",")) {
    const term = rawTerm.trim();
    if (!term) continue;
    const parts = term.split(":");
    if (parts.length === 1) {
      const exact = parseBound(parts[0]);
      if (exact == null) return { valid: false, match: () => true };
      predicates.push((step) => step === exact);
    } else if (parts.length === 2) {
      const start = parseBound(parts[0]);
      const stop = parseBound(parts[1]);
      if (start === undefined || stop === undefined) return { valid: false, match: () => true };
      predicates.push((step) => step >= (start ?? -Infinity) && step <= (stop ?? Infinity));
    } else if (parts.length === 3) {
      const start = parseBound(parts[0]);
      const stride = parseBound(parts[1]);
      const stop = parseBound(parts[2]);
      if (start === undefined || stride === undefined || stop === undefined || stride == null || stride <= 0) {
        return { valid: false, match: () => true };
      }
      const origin = start ?? 0;
      predicates.push(
        (step) =>
          step >= (start ?? -Infinity) &&
          step <= (stop ?? Infinity) &&
          Math.abs((step - origin) / stride - Math.round((step - origin) / stride)) < 1e-9,
      );
    } else {
      return { valid: false, match: () => true };
    }
  }
  if (!predicates.length) return all;
  return { valid: true, match: (step) => predicates.some((predicate) => predicate(step)) };
}

export function filterSteps(steps, expression, descending = false) {
  const parsed = parseStepFilter(expression);
  const filtered = parsed.valid ? steps.filter(parsed.match) : steps;
  return { valid: parsed.valid, steps: descending ? [...filtered].reverse() : filtered };
}

export function filterAndSortRows(rows, query, sort) {
  const terms = String(query ?? "")
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  const filtered = rows.filter((row) => terms.every((term) => row.name.toLowerCase().includes(term)));
  const numeric = (field, direction) => (left, right) => {
    const a = left[field];
    const b = right[field];
    if (a == null && b == null) return left.name.localeCompare(right.name);
    if (a == null) return 1;
    if (b == null) return -1;
    return direction * (a - b) || left.name.localeCompare(right.name);
  };
  const comparators = {
    name_asc: (left, right) => left.name.localeCompare(right.name, undefined, { numeric: true }),
    name_desc: (left, right) => right.name.localeCompare(left.name, undefined, { numeric: true }),
    latest_desc: numeric("latestValue", -1),
    latest_asc: numeric("latestValue", 1),
    delta_desc: numeric("delta", -1),
    delta_asc: numeric("delta", 1),
    step_desc: numeric("latestStep", -1),
    step_asc: numeric("latestStep", 1),
  };
  const valueSort = /^value:([-+]?\d+(?:\.\d+)?):(asc|desc)$/.exec(sort);
  if (valueSort) {
    const step = Number(valueSort[1]);
    const direction = valueSort[2] === "asc" ? 1 : -1;
    return filtered.sort((left, right) => {
      const a = left.points.get(step)?.value;
      const b = right.points.get(step)?.value;
      if (a == null && b == null) return left.name.localeCompare(right.name);
      if (a == null) return 1;
      if (b == null) return -1;
      return direction * (a - b) || left.name.localeCompare(right.name);
    });
  }
  return filtered.sort(comparators[sort] || comparators.name_asc);
}

export function nextColumnSort(currentSort, column) {
  const definitions = {
    name: { prefix: "name", initial: "asc" },
    latest: { prefix: "latest", initial: "desc" },
    delta: { prefix: "delta", initial: "desc" },
    at: { prefix: "step", initial: "desc" },
  };
  const definition = definitions[column];
  const prefix = definition?.prefix || column;
  const initial = definition?.initial || "desc";
  const asc = `${prefix}_asc`;
  const desc = `${prefix}_desc`;
  if (currentSort === asc) return desc;
  if (currentSort === desc) return asc;
  return `${prefix}_${initial}`;
}

export function columnSortDirection(sort, column) {
  const prefixes = { name: "name", latest: "latest", delta: "delta", at: "step" };
  const prefix = prefixes[column] || column;
  if (sort === `${prefix}_asc`) return "asc";
  if (sort === `${prefix}_desc`) return "desc";
  return null;
}

export function columnRanges(rows, steps) {
  const ranges = new Map();
  for (const step of steps) {
    const values = rows
      .map((row) => row.points.get(step)?.value)
      .filter(Number.isFinite)
      .sort((left, right) => left - right);
    if (!values.length) {
      ranges.set(step, { low: 0, center: 0, high: 0 });
      continue;
    }
    // Winsorize each visible column before coloring so a single broken run does
    // not flatten all useful differences among the remaining experiments.
    const trim = values.length >= 5 ? Math.max(1, Math.floor(values.length * 0.1)) : 0;
    const middle = Math.floor(values.length / 2);
    const center = values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
    ranges.set(step, {
      low: values[trim],
      center,
      high: values[values.length - 1 - trim],
    });
  }
  return ranges;
}

export function heatLevel(value, range, direction) {
  if (!Number.isFinite(value) || direction === "off") return null;
  let level = 0.5;
  if (value < range.center) {
    const span = range.center - range.low;
    level = span > 0 ? 0.5 - (0.5 * (range.center - value)) / span : 0;
  } else if (value > range.center) {
    const span = range.high - range.center;
    level = span > 0 ? 0.5 + (0.5 * (value - range.center)) / span : 1;
  }
  level = Math.max(0, Math.min(1, level));
  return direction === "lower" ? 1 - level : level;
}

export function formatValue(value) {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute === 0) return "0";
  if (absolute < 0.0001 || absolute >= 10000) return value.toExponential(3);
  if (absolute <= 1) return value.toFixed(4);
  if (absolute < 100) return value.toFixed(3);
  return value.toFixed(1);
}
