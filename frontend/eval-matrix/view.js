import {
  buildMatrix,
  columnSortDirection,
  columnRanges,
  defaultMetric,
  filterAndSortRows,
  filterSteps,
  formatValue,
  heatLevel,
  numericMetrics,
  nextColumnSort,
} from "./data.js";

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

export const controls = {
  metric: "",
  runFilter: "",
  stepFilter: "",
  sort: "name_asc",
  stepOrder: "asc",
  heat: "higher",
  scope: "selected",
};

export function shell(project) {
  return `
    <section class="op-eval-page">
      <header class="op-eval-hero">
        <div>
          <p class="op-eval-kicker">OPDLENS · COMPARISON SURFACE</p>
          <h1>Eval Matrix</h1>
          <p class="op-eval-subtitle">Loading scalar eval logs for <strong>${escapeHtml(project)}</strong>…</p>
        </div>
        <button class="op-eval-refresh" type="button" data-action="refresh" aria-label="Refresh data">
          <span aria-hidden="true">↻</span> Refresh
        </button>
      </header>
      <div class="op-eval-controls" aria-label="Eval matrix controls"></div>
      <div class="op-eval-color-legend">
        <span data-legend-bad>lower</span>
        <i aria-hidden="true"></i>
        <span data-legend-good>higher</span>
        <small>per step · median centered · 10% tails clipped</small>
      </div>
      <div class="op-eval-table-wrap"><div class="op-eval-empty">Loading results…</div></div>
      <div class="op-run-modal" hidden>
        <div class="op-run-modal-panel" role="dialog" aria-modal="true" aria-labelledby="op-run-modal-title">
          <header class="op-run-modal-header">
            <div>
              <p>RUN CONFIG</p>
              <h2 id="op-run-modal-title"></h2>
            </div>
            <div class="op-run-modal-actions">
              <button type="button" data-modal-action="copy">Copy JSON</button>
              <button type="button" class="op-run-modal-close" data-modal-action="close" aria-label="Close run config">×</button>
            </div>
          </header>
          <pre class="op-run-config"></pre>
        </div>
      </div>
    </section>
  `;
}

export function renderControls(host, metrics) {
  if (!metrics.includes(controls.metric)) controls.metric = defaultMetric(metrics);
  const metricOptions = metrics
    .map((metric) => `<option value="${escapeHtml(metric)}" ${metric === controls.metric ? "selected" : ""}>${escapeHtml(metric)}</option>`)
    .join("");
  const standardSorts = new Set(["name_asc", "name_desc", "latest_desc", "latest_asc", "delta_desc", "delta_asc", "step_desc", "step_asc"]);
  const customSort = standardSorts.has(controls.sort)
    ? ""
    : `<option value="${escapeHtml(controls.sort)}" selected>${escapeHtml(sortLabel(controls.sort))}</option>`;
  host.querySelector(".op-eval-controls").innerHTML = `
    <label class="op-field op-field-metric">
      <span>Metric</span>
      <select data-control="metric">${metricOptions}</select>
    </label>
    <label class="op-field">
      <span>Runs</span>
      <select data-control="scope">
        <option value="selected" ${controls.scope === "selected" ? "selected" : ""}>Sidebar selection</option>
        <option value="all" ${controls.scope === "all" ? "selected" : ""}>All project runs</option>
      </select>
    </label>
    <label class="op-field op-field-wide">
      <span>Run filter</span>
      <input data-control="runFilter" type="search" placeholder="e.g. round2 full" value="${escapeHtml(controls.runFilter)}">
    </label>
    <label class="op-field op-field-wide">
      <span>Steps</span>
      <input data-control="stepFilter" type="text" spellcheck="false" placeholder="0,50:300 or 0:50:300" value="${escapeHtml(controls.stepFilter)}">
    </label>
    <label class="op-field">
      <span>Sort runs</span>
      <select data-control="sort">
        ${customSort}
        <option value="name_asc" ${controls.sort === "name_asc" ? "selected" : ""}>Name A → Z</option>
        <option value="name_desc" ${controls.sort === "name_desc" ? "selected" : ""}>Name Z → A</option>
        <option value="latest_desc" ${controls.sort === "latest_desc" ? "selected" : ""}>Latest high → low</option>
        <option value="latest_asc" ${controls.sort === "latest_asc" ? "selected" : ""}>Latest low → high</option>
        <option value="delta_desc" ${controls.sort === "delta_desc" ? "selected" : ""}>Δ high → low</option>
        <option value="delta_asc" ${controls.sort === "delta_asc" ? "selected" : ""}>Δ low → high</option>
        <option value="step_desc" ${controls.sort === "step_desc" ? "selected" : ""}>Furthest step</option>
        <option value="step_asc" ${controls.sort === "step_asc" ? "selected" : ""}>Earliest latest step</option>
      </select>
    </label>
    <label class="op-field">
      <span>Step order</span>
      <select data-control="stepOrder">
        <option value="asc" ${controls.stepOrder === "asc" ? "selected" : ""}>Early → late</option>
        <option value="desc" ${controls.stepOrder === "desc" ? "selected" : ""}>Late → early</option>
      </select>
    </label>
    <label class="op-field">
      <span>Highlight</span>
      <select data-control="heat">
        <option value="higher" ${controls.heat === "higher" ? "selected" : ""}>Higher is better</option>
        <option value="lower" ${controls.heat === "lower" ? "selected" : ""}>Lower is better</option>
        <option value="off" ${controls.heat === "off" ? "selected" : ""}>Off</option>
      </select>
    </label>
  `;
}

function sortLabel(sort) {
  const match = /^value:([-+]?\d+(?:\.\d+)?):(asc|desc)$/.exec(sort);
  if (!match) return sort;
  return `Step ${match[1]} ${match[2] === "asc" ? "low → high" : "high → low"}`;
}

function sortHeader(label, column, extraClass = "") {
  const direction = columnSortDirection(controls.sort, column);
  const indicator = direction === "asc" ? "↑" : direction === "desc" ? "↓" : "↕";
  const ariaSort = direction === "asc" ? "ascending" : direction === "desc" ? "descending" : "none";
  return `<th class="${extraClass}" scope="col" aria-sort="${ariaSort}"><button type="button" class="op-sort-button" data-sort-column="${escapeHtml(column)}">${escapeHtml(label)}<span aria-hidden="true">${indicator}</span></button></th>`;
}

function syncSortSelect(host) {
  const select = host.querySelector('[data-control="sort"]');
  if (!select) return;
  let option = [...select.options].find((candidate) => candidate.value === controls.sort);
  if (!option) {
    option = document.createElement("option");
    option.value = controls.sort;
    option.textContent = sortLabel(controls.sort);
    select.prepend(option);
  }
  select.value = controls.sort;
}

function cell(row, step, range) {
  const point = row.points.get(step);
  if (!point) return '<td class="op-value op-missing">—</td>';
  const heat = heatLevel(point.value, range, controls.heat);
  let style = "";
  if (heat != null) {
    const strength = Math.abs(heat - 0.5) * 2;
    const tone = heat >= 0.5 ? "#087f72" : "#c64b40";
    style = ` style="--op-tone:${tone};--op-tone-fill:${(strength * 34).toFixed(2)}%;--op-tone-line:${(strength * 88).toFixed(2)}%;--op-tone-text:${(strength * 24).toFixed(2)}%"`;
  }
  const time = point.timestamp ? new Date(point.timestamp).toLocaleString() : "unknown time";
  return `<td class="op-value"${style} title="${escapeHtml(row.name)} · step ${step} · ${point.value} · ${escapeHtml(time)}">${formatValue(point.value)}</td>`;
}

export function renderTable(host, state) {
  const { allRuns, selectedRuns, runLogs } = state;
  const metrics = numericMetrics(runLogs);
  const availableRuns = controls.scope === "all" ? allRuns : selectedRuns;
  const matrix = buildMatrix(runLogs, availableRuns, controls.metric);
  const rows = filterAndSortRows(matrix.rows, controls.runFilter, controls.sort);
  const stepResult = filterSteps(matrix.steps, controls.stepFilter, controls.stepOrder === "desc");
  const steps = stepResult.steps;
  const ranges = columnRanges(rows, steps);
  const tableWrap = host.querySelector(".op-eval-table-wrap");
  const legend = host.querySelector(".op-eval-color-legend");
  legend.hidden = controls.heat === "off";
  legend.querySelector("[data-legend-bad]").textContent = controls.heat === "lower" ? "higher" : "lower";
  legend.querySelector("[data-legend-good]").textContent = controls.heat === "lower" ? "lower" : "higher";
  host.querySelector('[data-control="stepFilter"]')?.classList.toggle("op-invalid", !stepResult.valid);
  host.querySelector(".op-eval-subtitle").innerHTML = `
    <strong>${rows.length}</strong> of ${availableRuns.length} runs ·
    <strong>${steps.length}</strong> eval steps ·
    <code>${escapeHtml(controls.metric || "no metric")}</code>
  `;

  if (!metrics.length) {
    tableWrap.innerHTML = '<div class="op-eval-empty">No scalar metrics were found for these runs.</div>';
    return;
  }
  if (!availableRuns.length) {
    tableWrap.innerHTML = '<div class="op-eval-empty">Select runs in the Trackio sidebar, or switch Runs to “All project runs”.</div>';
    return;
  }
  if (!rows.length) {
    tableWrap.innerHTML = '<div class="op-eval-empty">No run names match this filter.</div>';
    return;
  }
  if (!steps.length) {
    tableWrap.innerHTML = '<div class="op-eval-empty">This metric has no values at the requested steps.</div>';
    return;
  }

  const colors = state.colors;
  const head = steps.map((step) => sortHeader(`step ${step}`, `value:${step}`, "op-step")).join("");
  const body = rows
    .map((row) => {
      const color = colors.get(row.name) || "#f97316";
      return `
        <tr>
          <th class="op-run" scope="row" title="Open ${escapeHtml(row.name)} config">
            <button type="button" class="op-run-button" data-run-id="${escapeHtml(row.id)}">
              <i style="background:${escapeHtml(color)}"></i><span>${escapeHtml(row.name)}</span>
            </button>
          </th>
          <td class="op-summary op-latest">${formatValue(row.latestValue)}</td>
          <td class="op-summary ${row.delta > 0 ? "op-positive" : row.delta < 0 ? "op-negative" : ""}">${row.delta > 0 ? "+" : ""}${formatValue(row.delta)}</td>
          <td class="op-summary">${row.latestStep ?? "—"}</td>
          ${steps.map((step) => cell(row, step, ranges.get(step))).join("")}
        </tr>
      `;
    })
    .join("");
  tableWrap.innerHTML = `
    <table class="op-eval-table">
      <thead><tr>
        ${sortHeader("Run", "name", "op-run op-corner")}
        ${sortHeader("Latest", "latest", "op-summary-head")}
        ${sortHeader("Δ", "delta", "op-summary-head")}
        ${sortHeader("At", "at", "op-summary-head")}
        ${head}
      </tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function openRunConfig(host, state, runId) {
  const run = state.allRuns.find((candidate) => String(candidate.id || candidate.name) === String(runId));
  if (!run) return;
  const config = state.runConfigs[run.id] ?? state.runConfigs[run.name] ?? {};
  const modal = host.querySelector(".op-run-modal");
  modal.querySelector("h2").textContent = run.name;
  modal.querySelector("pre").textContent = JSON.stringify(config, null, 2);
  modal.hidden = false;
  modal.querySelector('[data-modal-action="close"]')?.focus();
}

function closeRunConfig(host) {
  host.querySelector(".op-run-modal")?.setAttribute("hidden", "");
}

async function copyRunConfig(host) {
  const button = host.querySelector('[data-modal-action="copy"]');
  const json = host.querySelector(".op-run-config")?.textContent || "{}";
  try {
    await navigator.clipboard.writeText(json);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = json;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = "Copy JSON"; }, 1400);
}

export function bindControls(host, state, callbacks) {
  host.addEventListener("input", (event) => {
    const name = event.target?.dataset?.control;
    if (!name || !(name in controls)) return;
    controls[name] = event.target.value;
    if (name === "scope") callbacks.reload();
    else renderTable(host, state);
  });
  host.addEventListener("change", (event) => {
    const name = event.target?.dataset?.control;
    if (!name || !(name in controls)) return;
    controls[name] = event.target.value;
    if (name === "scope") callbacks.reload();
    else renderTable(host, state);
  });
  host.querySelector('[data-action="refresh"]')?.addEventListener("click", callbacks.refresh);
  host.addEventListener("click", (event) => {
    const sort = event.target?.closest?.("[data-sort-column]");
    if (sort) {
      controls.sort = nextColumnSort(controls.sort, sort.dataset.sortColumn);
      syncSortSelect(host);
      renderTable(host, state);
      return;
    }
    const run = event.target?.closest?.("[data-run-id]");
    if (run) {
      openRunConfig(host, state, run.dataset.runId);
      return;
    }
    const modalAction = event.target?.closest?.("[data-modal-action]")?.dataset.modalAction;
    if (modalAction === "close" || event.target?.classList?.contains("op-run-modal")) closeRunConfig(host);
    if (modalAction === "copy") copyRunConfig(host);
  });
  host.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeRunConfig(host);
  });
}
